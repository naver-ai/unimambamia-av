# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# MambaMia2 backbone shared by the audio compressors (see audio_compressor.py):
#   version "v0"  : causal Mamba2                              (UniMamba)
#   version "v04" : causal Mamba2 + gated pooling aggregation  (UniMambaMia)
#   version "v2"  : bi-directional Mamba2                      (BiMamba)
#
# Modified from https://github.com/naver-ai/mambamia/blob/main/MambaMia/modeling_mambamia2.py,
# which is modified from
# https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/mamba2/modeling_mamba2.py

# coding=utf-8
# Copyright 2024 state-spaces/mamba2 org and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MAMBAMIA2 model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.utils.import_utils import is_causal_conv1d_available, is_mamba_2_ssm_available

if __name__ == "__main__":
    import sys
    from os import path

    print(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from configuration_mambamia2 import MambaMia2Config
else:
    try:
        from .configuration_mambamia2 import MambaMia2Config
    except:
        import sys
        from os import path

        print(path.dirname(path.dirname(path.abspath(__file__))))
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from configuration_mambamia2 import MambaMia2Config


import torch.nn.functional as F
from einops import rearrange

logger = logging.get_logger(__name__)


if is_mamba_2_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
else:
    selective_state_update = None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all((selective_state_update, causal_conv1d_fn, causal_conv1d_update))

_CHECKPOINT_FOR_DOC = "mistralai/mamba-codestral-7B-v0.1"
_CONFIG_FOR_DOC = "MambaMia2Config"


# Helper methods for segment sum computation


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3])


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


class MambaMia2Cache:
    """
    Arguments:
        config: MambaMia2Config
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype
        conv_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, conv_kernel_size]
        ssm_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, ssm_state_size]
    """

    def __init__(
        self, config: MambaMia2Config, batch_size: int, dtype: torch.dtype = torch.float16, device: Optional[str] = None
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * config.hidden_size)

        self.conv_states = {
            i: torch.zeros(
                batch_size,
                self.intermediate_size + 2 * config.n_groups * config.state_size,
                self.conv_kernel_size,
                device=device,
                dtype=dtype,
            )
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, config.num_heads, config.head_dim, config.state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

    def update_conv_state(
        self, layer_idx: int, new_conv_state: torch.Tensor, cache_position: torch.LongTensor
    ) -> torch.Tensor:
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(conv_state.device)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def reset(self):
        self.conv_states.zero_()
        self.ssm_states.zero_()


class MambaRMSNormGated(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


class MambaMia2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: MambaMia2Config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.norm_before_gate = config.norm_before_gate
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(
            self.hidden_size,
            projection_size,
            bias=config.use_bias,
        )
        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        ##################################################################################
        self.version = config.version

        if config.version == "v2":
            self.conv1d_b = nn.Conv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                bias=config.use_conv_bias,
                kernel_size=config.conv_kernel,
                groups=self.conv_dim,
                padding=config.conv_kernel - 1,
            )
            self.dt_bias_b = nn.Parameter(torch.ones(self.num_heads))
            A_b = torch.arange(1, self.num_heads + 1)
            self.A_b_log = nn.Parameter(torch.log(A_b))
            self.A_b_log._no_weight_decay = True
            self.norm_b = MambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)
            self.D_b = nn.Parameter(torch.ones(self.num_heads))
            self.D_b._no_weight_decay = True
        ##################################################################################

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, causal_conv1d_fn, causal_conv1d_update)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[MambaMia2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # set up dimensions for reshapes later

        dtype = hidden_states.dtype  # origianl dtype
        batch_size, seq_len, _ = hidden_states.shape
        # groups_time_state_size = self.n_groups * self.ssm_state_size
        # d_to_remove = 2 * self.intermediate_size + 2 * self.n_groups * self.ssm_state_size + self.num_heads

        assert cache_params is None
        # if no cache is found, calling the kernel

        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)
        dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

        if self.version in ["v0", "v04"]:

            outputs = mamba_split_conv1d_scan_combined(
                projected_states,  # (batch, seq_len, dim)
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.dt_bias,
                -torch.exp(self.A_log.float()),  # (num_heads) or (intermediate_size, state_size)
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=None,  # was seq_idx
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.variance_epsilon,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=self.head_dim,
                ngroups=self.n_groups,
                norm_before_gate=self.norm_before_gate,
                return_final_states=False,
                **dt_limit_kwargs,
            )

            return outputs.to(dtype)

        if self.version == "v2":
            # bi-directional
            out_f = mamba_split_conv1d_scan_combined(
                projected_states,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.dt_bias,
                -torch.exp(self.A_log.float()),  # (num_heads) or (intermediate_size, state_size)
                D=self.D,
                chunk_size=self.chunk_size,
                seq_idx=None,  # was seq_idx
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.variance_epsilon,
                outproj_weight=None,
                outproj_bias=None,
                headdim=self.head_dim,
                ngroups=self.n_groups,
                norm_before_gate=self.norm_before_gate,
                return_final_states=False,
                **dt_limit_kwargs,
            )
            out_b = mamba_split_conv1d_scan_combined(
                projected_states.flip([1]),
                self.conv1d_b.weight.squeeze(1),
                self.conv1d_b.bias,
                self.dt_bias_b,
                -torch.exp(self.A_b_log.float()),  # (num_heads) or (intermediate_size, state_size)
                D=self.D_b,
                chunk_size=self.chunk_size,
                seq_idx=None,  # was seq_idx
                activation=self.activation,
                rmsnorm_weight=self.norm_b.weight,
                rmsnorm_eps=self.norm_b.variance_epsilon,
                outproj_weight=None,
                outproj_bias=None,
                headdim=self.head_dim,
                ngroups=self.n_groups,
                norm_before_gate=self.norm_before_gate,
                return_final_states=False,
                **dt_limit_kwargs,
            )  # (batch, seq_len, dim)
            out = F.linear(
                out_f + out_b.flip([1]), self.out_proj.weight, self.out_proj.bias if self.use_bias else None
            )
            return out.to(dtype)

        raise ValueError(
            f"Unsupported MambaMia2 version {self.version}. Supported versions are: ['v0', 'v04', 'v2']"
        )

    def forward(
        self,
        hidden_states,
        cache_params: Optional[MambaMia2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        assert is_fast_path_available and "cuda" in self.in_proj.weight.device.type
        return self.cuda_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)


class MambaMia2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaMia2RMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaMia2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32

        self.version = config.version
        self.mambamia_chunk_size = config.mambamia_chunk_size

        self.norm = MambaMia2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMia2Mixer(config, layer_idx=layer_idx)

        assert self.version in ["v0", "v04", "v2"], f"Unsupported MambaMia2 version: {self.version}"

        if self.version == "v04":
            # Gated pooling aggregation (MambaMia): each query token is updated with a learned,
            # softmax-weighted average of its block, gated against the query itself.
            self.drop = nn.Dropout(p=0.1)
            init_scale = getattr(config, "mambamia_init_scale", 1e-3)

            # (1) Per-frame weight prediction (shape=(bsz, n_chunk, 10)) -> softmax -> weighted sum
            self.weight_fc = nn.Linear(config.hidden_size, self.mambamia_chunk_size)
            nn.init.zeros_(self.weight_fc.bias)
            with torch.no_grad():
                self.weight_fc.weight.mul_(init_scale)

            # (2) Query vs aggregator gating (scaling factor)
            self.gate_fc = nn.Linear(config.hidden_size, 1)  # (bsz, n_chunk, 1)
            nn.init.zeros_(self.gate_fc.bias)
            with torch.no_grad():
                self.gate_fc.weight.mul_(init_scale)

    def forward(
        self,
        hidden_states,
        cache_params: Optional[MambaMia2Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if self.version == "v04":
            # Weighted average + query gating
            assert hidden_states.dim() == 3, f"[v04] hidden_states.dim()={hidden_states.dim()} != 3"
            bsz, seq_len, hidden_dim = hidden_states.shape
            mambamia_chunk_size = self.mambamia_chunk_size
            chunk_with_query = mambamia_chunk_size + 1  # = 11
            if seq_len % chunk_with_query != 0:
                raise ValueError(
                    f"[v04] seq_len={seq_len} is not divisible by (mambamia_chunk_size+1)={chunk_with_query}."
                )
            n_chunk = seq_len // chunk_with_query

            # (bsz, n_chunk, 11, hidden_dim)
            hidden_4d = hidden_states.view(bsz, n_chunk, chunk_with_query, hidden_dim)

            frames_10 = hidden_4d[:, :, :mambamia_chunk_size, :]  # (bsz, n_chunk, 10, hidden_dim)
            queries = hidden_4d[:, :, mambamia_chunk_size, :]  # (bsz, n_chunk, hidden_dim)

            # (C'-1) Compute weight for each frame in frames_10
            #        queries -> weight_fc -> (bsz, n_chunk, 10)
            w_in = self.drop(queries)
            raw_weights = self.weight_fc(w_in)
            alpha = torch.softmax(raw_weights, dim=-1)  # (bsz, n_chunk, 10)

            # (C'-2) weighted average: aggregator
            aggregator = (frames_10 * alpha.unsqueeze(-1)).sum(dim=2)  # (bsz, n_chunk, hidden_dim)

            # (C'-3) Query vs aggregator gating (scalar)
            gating_in = self.drop(queries)
            gating = torch.sigmoid(self.gate_fc(gating_in))  # (bsz, n_chunk, 1)
            epsilon = 0.01
            gating = gating * (1 - 2 * epsilon) + epsilon  # [0.01, 0.99]

            gating_broad = gating.expand(-1, -1, hidden_dim)
            aggregator = aggregator * gating_broad
            queries = queries * (1 - gating_broad)
            queries_new = queries + aggregator
            # Note: The following in-place assignment can potentially interfere with autograd if hidden_4d is a view of the original tensor.
            # To avoid issues with gradient computation, use clone() to ensure hidden_4d is not a view.
            hidden_4d = hidden_4d.clone()
            # (C'-4) Update query position and restore
            hidden_4d[:, :, mambamia_chunk_size, :] = queries_new
            hidden_states = hidden_4d.view(bsz, seq_len, hidden_dim)

        hidden_states = self.mixer(
            hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + residual
        return hidden_states


class MambaMia2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MambaMia2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["MambaMia2Block"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MambaMia2Mixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            if self.config.version == "v2":
                module.A_b_log._no_weight_decay = True
                module.D_b._no_weight_decay = True

            dt = torch.exp(
                torch.rand(self.config.num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)

            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)


@dataclass
# Copied from transformers.models.mamba.modeling_mamba.MambaOutput with MAMBA->MAMBA2,Mamba->MambaMia2
class MambaMia2Output(ModelOutput):
    """
    Class for the MAMBA2 model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`MambaMia2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaMia2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# Copied from transformers.models.mamba.modeling_mamba.MambaCausalLMOutput with Mamba->MambaMia2
class MambaMia2CausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`MambaMia2Cache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaMia2Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


MAMBA2_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MambaMia2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MAMBA2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            Indices of input sequence tokens in the vocabulary.

            If `cache_params.seqlen_offset>0`, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        cache_params (`MambaMia2Cache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MAMBA2 Model transformer outputting raw hidden-states without any specific head on top.",
    MAMBA2_START_DOCSTRING,
)
class MambaMia2Model(MambaMia2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MambaMia2Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaMia2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(MAMBA2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MambaMia2Output,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[MambaMia2Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, MambaMia2Output]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if use_cache:
            if cache_params is None:
                cache_params = MambaMia2Cache(
                    self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                cache_position = torch.arange(0, self.config.conv_kernel, device=inputs_embeds.device)
            elif cache_position is None:
                # cases when we do manual forward instead of using `model.generate` which will initiate
                # `cache_position` and makes sure it is not None, throw error here instead of doing some
                # hack to conjecture the current cache position
                raise ValueError(
                    "You have to specify the `cache_position` manually when `use_cache=True` and `cache_params` is passed, "
                    "you don't have to pass a `cache_params` if you are in prefilling stage because in that case it will "
                    "be initialized for you automatically"
                )
        else:
            cache_params = None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, cache_params, cache_position, attention_mask
                )
            else:
                hidden_states = mixer_block(
                    hidden_states,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaMia2Output(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )
