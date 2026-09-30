# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Modified from https://github.com/naver-ai/mambamia/blob/main/MambaMia/model_builder.py,
# which is modified from
# https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/model/builder.py
# Below is the original copyright:
#
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import warnings

import torch
from transformers import AutoTokenizer

from llava.model import LlavaQwenForCausalLM

_TORCH_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_pretrained_model(
    model_path,
    model_base=None,
    model_name=None,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    torch_dtype="float16",
    **kwargs,
):
    """Load a UniMambaMia-AV checkpoint (Qwen2 backbone, vision tower, audio tower, projectors).

    Returns:
        ``(tokenizer, model, image_processor, audio_processor, context_len)``. ``audio_processor``
        is ``None`` for checkpoints trained without an audio tower.
    """
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if torch_dtype not in _TORCH_DTYPES:
        warnings.warn(f"Unknown torch_dtype: {torch_dtype}, using float16 instead.")
        torch_dtype = "float16"
    torch_dtype = _TORCH_DTYPES[torch_dtype]
    kwargs["torch_dtype"] = torch_dtype

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # delay_load=False: the towers are built together with the LLM so that their weights are read
    # from the checkpoint rather than re-downloaded and randomly re-initialized.
    model = LlavaQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, delay_load=False, **kwargs)

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    mm_projector = model.get_mm_projector()
    if mm_projector is None:
        raise ValueError("The checkpoint has no vision projector; it was probably saved with delay_load=True.")
    if device_map != "auto":
        vision_tower.to(device=device_map, dtype=torch_dtype)
        mm_projector.to(device=device_map, dtype=torch_dtype)
    image_processor = vision_tower.image_processor

    audio_processor = None
    audio_tower = model.get_audio_tower()
    if audio_tower is not None:
        if not audio_tower.is_loaded:
            audio_tower.load_model(device_map=device_map)
        if device_map != "auto":
            audio_tower.to(device=device_map, dtype=torch_dtype)
            mm_audio_projector = model.get_mm_audio_projector()
            if mm_audio_projector is not None:
                mm_audio_projector.to(device=device_map, dtype=torch_dtype)
        audio_processor = audio_tower.audio_processor

    context_len = getattr(model.config, "max_sequence_length", 2048)

    # The vision tower keeps only the layers up to mm_vision_select_layer.
    if hasattr(vision_tower, "remove_unused_layers"):
        vision_tower.remove_unused_layers()

    return tokenizer, model, image_processor, audio_processor, context_len
