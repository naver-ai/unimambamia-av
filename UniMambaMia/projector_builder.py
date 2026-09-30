# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Modified from https://github.com/naver-ai/mambamia/blob/main/MambaMia/projector_builder.py,
# which is modified from
# https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/model/multimodal_projector/builder.py

"""Builders for the vision projector and the audio compressor.

Vision (``config.mm_projector_type``):
    ``mlp{N}x_gelu``  N-layer MLP (the released model uses ``mlp2x_gelu``)
    ``linear``        single linear layer
    ``identity``      no projection

Audio (``config.mm_audio_projector_type``), compared in the paper under 25x compression:
    ``audio_projector_mambamia2uni``  causal Mamba2 + gated attention (UniMambaMia, released model)
    ``audio_projector_mamba2uni``     causal Mamba2 (UniMamba)
    ``audio_projector_mamba2bi``      bi-directional Mamba2 (BiMamba)
    ``audio_projector_resampler``     bi-directional transformer with learnable queries (Resampler)
    ``audio_{R}patchwise_mlp{N}x_gelu``  MLP followed by average pooling every R tokens (Avg Pool)
    ``audio_mlp{N}x_gelu``            MLP without compression

The compression ratio ``R`` of the Mamba-based compressors is read from
``config.audio_projector_mode`` (``per_from1to1frames_{R}patchwise_{R}tokperframe``); the Avg Pool
baseline encodes it in the projector name and pools in ``LlavaMetaForCausalLM.encode_audios``.
"""

import re

import torch.nn as nn

from .audio_compressor import AudioCompressor, AudioCompressorConfig

# projector name -> audio compressor backbone
AUDIO_COMPRESSORS = {
    "audio_projector_mambamia2uni": "mambamia2uni",
    "audio_projector_mamba2uni": "mamba2uni",
    "audio_projector_mamba2bi": "mamba2bi",
    "audio_projector_resampler": "resampler",
    "audio_projector_bigptneox": "resampler",  # name used while training the Resampler baseline
}


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_loaded = False

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


def _mlp(input_size, hidden_size, depth):
    modules = [nn.Linear(input_size, hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*modules)


def build_mm_projector(config, projector_type=None, delay_load=False):
    """Build the module that maps encoder features into the LLM embedding space.

    Args:
        config: the LLaVA model config.
        projector_type: ``config.mm_projector_type`` (vision) or ``config.mm_audio_projector_type``
            (audio). Defaults to the vision projector.
        delay_load: return ``None`` instead of building; the module is built later by
            ``initialize_vision_modules``.
    """
    if delay_load:
        return None

    if projector_type is None:
        projector_type = getattr(config, "mm_projector_type", "identity")

    # ---------------------------------------------------------------- audio
    if projector_type in AUDIO_COMPRESSORS:
        mode = getattr(config, "audio_projector_mode", None)
        if mode is None:
            raise ValueError("config.audio_projector_mode is required for Mamba-based audio compressors.")
        match = re.search(r"(\d+)patchwise", mode)
        if not match:
            raise ValueError(f"audio_projector_mode must contain '{{R}}patchwise' (compression ratio), got {mode!r}.")
        return AudioCompressor(
            AudioCompressorConfig(
                backbone=AUDIO_COMPRESSORS[projector_type],
                block_size=int(match.group(1)),
                input_size=config.mm_audio_hidden_size,
                output_size=config.hidden_size,
                num_hidden_layers=config.audio_projector_num_layers,
                hidden_size=getattr(config, "audio_projector_hidden_size", None) or 3072,
                expand=getattr(config, "audio_projector_expand", None) or 2.0,
                init_scale=getattr(config, "audio_projector_init_scale", None) or 1e-3,
            )
        )

    # Avg Pool baseline and the uncompressed MLP: the pooling itself happens in encode_audios().
    audio_mlp = re.match(r"^audio_(?:\d+patchwise_)?mlp(\d+)x_gelu$", projector_type)
    if audio_mlp:
        return _mlp(config.mm_audio_hidden_size, config.hidden_size, int(audio_mlp.group(1)))

    # --------------------------------------------------------------- vision
    if projector_type == "identity":
        return IdentityMap()
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    mlp = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp:
        return _mlp(config.mm_hidden_size, config.hidden_size, int(mlp.group(1)))

    raise ValueError(f"Unknown projector type: {projector_type}")
