# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Modified from https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/model/multimodal_encoder/builder.py

import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, WhisperFeatureExtractor

from .clip_encoder import CLIPVisionTower, SigLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)

    if "siglip" in vision_tower.lower():
        return SigLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif (
        is_absolute_path_exists
        or vision_tower.startswith("openai")
        or vision_tower.startswith("laion")
        or "ShareGPT4V" in vision_tower
    ):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


class AudioTower(nn.Module):
    """Frozen speech/audio encoder.

    We use the Whisper-style encoder of Qwen2-Audio (encoder only, not the full audio LLM),
    released as ``gwkrsrch2/qwen2-audio-encoder-from-qwen2-audio-7b-instruct``.
    Raw 16 kHz waveforms are converted to 128-bin log-Mel spectrograms by
    ``WhisperFeatureExtractor`` (30 s windows, 3000 frames) and encoded into 25 Hz features
    (750 tokens per 30 s window, d_model=1280).
    """

    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.audio_tower_name = audio_tower

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.audio_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.audio_tower_name))
            return
        print(f"This will load audio_tower: {self.audio_tower_name}")
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(self.audio_tower_name)
        self.model = AutoModel.from_pretrained(self.audio_tower_name, device_map=device_map)
        self.model.requires_grad_(False)
        self.model.eval()
        self.is_loaded = True

    @torch.no_grad()  # the audio encoder is always frozen
    def forward(self, audio, attention_mask):
        return self.model(audio, attention_mask=attention_mask)

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.model.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.d_model


def build_audio_tower(audio_tower_cfg, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
    print(f"This will load audio_tower: {audio_tower}")
    return AudioTower(audio_tower, args=audio_tower_cfg, **kwargs)
