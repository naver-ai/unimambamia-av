# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Adapted from the video projector of
# https://github.com/naver-ai/mambamia/blob/main/MambaMia/video_projectors.py
#
# `BidirectionalGPTNeoX` subclasses the GPT-NeoX implementation of
# https://github.com/huggingface/transformers (Apache-2.0) and disables its causal mask.

"""Periodic-query audio compressor.

The speech/audio encoder emits features at 25 Hz, so one hour of video yields about 90K audio
tokens. This module reduces them by a factor ``R``: the encoder features are split into blocks of
``R`` tokens, one shared learnable query is appended to every block, the whole sequence is passed
through a small Mamba2 network, and only the outputs at the query positions are kept. With the
default ``R = 25`` the LLM receives one audio token per second of video.

Four compressor variants share this interface (``backbone`` argument):

=================  =========================================================================
``backbone``       Description
=================  =========================================================================
``mambamia2uni``   Causal Mamba2 with gated pooling aggregation (UniMambaMia, released model)
``mamba2uni``      Causal Mamba2 (UniMamba)
``mamba2bi``       Bi-directional Mamba2 (BiMamba)
``resampler``      Bi-directional transformer with learnable queries (Resampler)
=================  =========================================================================

Only the causal variants are compatible with streaming inference, where audio arrives
incrementally alongside video frames.
"""

import math

import torch
from torch import nn
from transformers import GPTNeoXConfig, GPTNeoXModel, PretrainedConfig, PreTrainedModel

if __name__ == "__main__":
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from configuration_mambamia2 import MambaMia2Config
    from modeling_mambamia2 import MambaMia2Model
else:
    from .configuration_mambamia2 import MambaMia2Config
    from .modeling_mambamia2 import MambaMia2Model


# Mamba-based compressors: name -> MambaMia2 backbone version
MAMBA_VERSIONS = {
    "mambamia2uni": "v04",
    "mamba2uni": "v0",
    "mamba2bi": "v2",
}
BACKBONES = sorted(MAMBA_VERSIONS) + ["resampler"]


class BidirectionalGPTNeoX(GPTNeoXModel):
    """GPT-NeoX encoder without the causal mask, used for the Resampler baseline.

    In the periodic-query design the queries summarize the whole clip, so the Resampler baseline
    runs bidirectionally. Inputs are single unpadded sequences, so no attention mask is needed.
    """

    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            layer.attention.is_causal = False

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
        return None


class AudioCompressorConfig(PretrainedConfig):
    """Configuration of :class:`AudioCompressor`.

    Args:
        backbone: One of ``mambamia2uni``, ``mamba2uni``, ``mamba2bi``, ``resampler``.
        block_size: Compression ratio ``R``; one query is inserted every ``R`` encoder tokens.
        input_size: Width of the audio encoder features (1280 for the Qwen2-Audio encoder).
        output_size: Width of the LLM embedding space.
        num_hidden_layers: Number of layers in the backbone.
        hidden_size: Width of the backbone.
        expand: Expansion ratio of the Mamba2 backbone (ignored by ``resampler``).
        init_scale: Initialization scale of the gated-attention projections (``mambamia2uni`` only).
    """

    model_type = "audio_compressor"

    def __init__(
        self,
        backbone="mambamia2uni",
        block_size=25,
        input_size=1280,
        output_size=3584,
        num_hidden_layers=2,
        hidden_size=3072,
        expand=2.0,
        init_scale=1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.block_size = block_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.expand = expand
        self.init_scale = init_scale


class AudioCompressor(PreTrainedModel):
    """Compresses audio encoder features by a factor of ``config.block_size``."""

    config_class = AudioCompressorConfig
    base_model_prefix = "audio_compressor"

    def __init__(self, config: AudioCompressorConfig):
        super().__init__(config)

        if config.backbone not in BACKBONES:
            raise ValueError(f"Unknown audio compressor backbone {config.backbone!r}; expected one of {BACKBONES}.")

        if config.backbone == "resampler":
            self.model = BidirectionalGPTNeoX(
                GPTNeoXConfig(
                    vocab_size=0,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=12,
                    intermediate_size=int(config.hidden_size * 1.5),
                    max_position_embeddings=300000,  # a one-hour clip is ~94K positions at R=25
                    use_cache=False,
                )
            )
        else:
            head_dim = 64
            self.model = MambaMia2Model(
                MambaMia2Config(
                    vocab_size=0,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    head_dim=head_dim,
                    num_heads=int(config.hidden_size * config.expand) // head_dim,
                    n_groups=1,
                    expand=config.expand,
                    use_cache=False,
                    version=MAMBA_VERSIONS[config.backbone],
                    mambamia_chunk_size=config.block_size,
                    mambamia_init_scale=config.init_scale,
                    residual_in_fp32=False,
                )
            )

        embed_std = 1 / math.sqrt(config.hidden_size)
        self.query_token = nn.Parameter(torch.randn(config.hidden_size, dtype=self.dtype) * embed_std)

        if config.input_size == config.hidden_size:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(config.input_size, config.hidden_size)
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)

        if config.hidden_size == config.output_size:
            self.output_proj = None
        else:
            self.output_proj = nn.Linear(config.hidden_size, config.output_size)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        self.post_init()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compress the audio features of one clip.

        Args:
            features: ``(num_blocks, block_size, input_size)``. Callers pad the encoder output to a
                multiple of ``block_size`` and reshape it; see ``LlavaMetaForCausalLM.encode_audios``.

        Returns:
            ``(num_blocks, output_size)``: one token per block, in temporal order.
        """
        num_blocks, block_size, input_size = features.shape
        if block_size != self.config.block_size:
            raise ValueError(f"Expected blocks of {self.config.block_size} tokens, got {block_size}.")
        if input_size != self.config.input_size:
            raise ValueError(f"Expected features of width {self.config.input_size}, got {input_size}.")

        hidden = self.input_proj(features)  # (num_blocks, block_size, hidden_size)

        # Append the shared query to every block and flatten the clip into one sequence, so that the
        # state-space model carries context across block boundaries.
        queries = self.query_token.view(1, 1, -1).expand(num_blocks, 1, -1)
        hidden = torch.cat([hidden, queries], dim=1).reshape(1, -1, hidden.size(-1))

        hidden = self.model(inputs_embeds=hidden).last_hidden_state.squeeze(0)

        # Keep the query positions only: index block_size within each (block_size + 1) group.
        hidden = hidden.view(num_blocks, block_size + 1, -1)[:, block_size, :]

        if self.output_proj is not None:
            hidden = self.output_proj(hidden.to(self.output_proj.weight.dtype))
        return hidden
