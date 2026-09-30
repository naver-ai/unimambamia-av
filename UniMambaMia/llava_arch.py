# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Modified from https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/model/llava_arch.py
# and https://github.com/naver-ai/mambamia/blob/main/MambaMia/llava_arch.py
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


from abc import ABC, abstractmethod

import re
import math
import torch
import torch.nn as nn
import numpy as np

from .multimodal_encoder.builder import build_vision_tower, build_audio_tower
from .multimodal_projector.audio_compressor import AudioCompressor
from .multimodal_projector.builder import build_mm_projector

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)



class LlavaMetaModel:

    def __init__(self, config, delay_load=True):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower") and config.mm_vision_tower is not None:
            print(f"delay_load: {delay_load}")
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_mm_projector(
                config, projector_type=config.mm_projector_type, delay_load=delay_load
            )

            if 'video' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newframe = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

        # Audio tower (frozen speech/audio encoder) + audio compressor
        if getattr(config, "mm_audio_tower", None) is not None:
            print(f"audio delay_load: {delay_load}")
            self.audio_tower = build_audio_tower(config, delay_load=True)
            # d_model of the Qwen2-Audio encoder; read from the (delay-loaded) config.
            config.mm_audio_hidden_size = self.audio_tower.hidden_size
            self.mm_audio_projector = build_mm_projector(
                config, projector_type=config.mm_audio_projector_type, delay_load=delay_load
            )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_audio_tower(self):
        audio_tower = getattr(self, 'audio_tower', None)
        if type(audio_tower) is list:
            audio_tower = audio_tower[0]
        return audio_tower

    def get_mm_projector(self):
        mm_projector = getattr(self, 'mm_projector', None)
        return mm_projector

    def get_mm_audio_projector(self):
        mm_audio_projector = getattr(self, 'mm_audio_projector', None)
        return mm_audio_projector

    def set_mm_projector(self, mm_projector):
        assert getattr(self, 'mm_projector', None) is None
        self.mm_projector = mm_projector

    def set_mm_audio_projector(self, mm_audio_projector):
        assert getattr(self, 'mm_audio_projector', None) is None
        self.mm_audio_projector = mm_audio_projector

    def initialize_vision_modules(self, model_args, fsdp=None, deepspeed=True):
        """Initialize vision (and, if ``model_args.audio_tower`` is set, audio) modules."""
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        if not hasattr(self, 'image_newframe') and 'video' in mm_patch_merge_type:
            self.image_newframe = nn.Parameter(
                torch.randn(self.config.hidden_size, dtype=self.dtype) * (1 / np.sqrt(float(self.config.hidden_size)))
            )

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args, delay_load=False)

            self.vision_tower = vision_tower
        else:
            if (
                fsdp is not None and len(fsdp) > 0 and not deepspeed and isinstance(self.vision_tower, list)
            ):
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        self.mm_projector = build_mm_projector(self.config, projector_type=self.config.mm_projector_type)

        print(f"mm_projector parameter size: {sum(p.numel() for p in self.mm_projector.parameters())}")

        for p in self.mm_projector.parameters():
            p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu', weights_only=True)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, 'mm_projector'), strict=not model_args.ignore_mismatched_sizes
            )

        # ------------------------------------------------------------------
        # Audio tower + audio compressor
        # ------------------------------------------------------------------
        audio_tower = getattr(model_args, 'audio_tower', None)
        self.config.mm_audio_tower = audio_tower
        self.config.mm_audio_projector_type = getattr(model_args, 'mm_audio_projector_type', 'linear')
        self.config.audio_projector_mode = getattr(model_args, 'audio_projector_mode', None)
        self.config.audio_projector_num_layers = getattr(model_args, 'audio_projector_num_layers', None)
        self.config.audio_projector_init_scale = getattr(model_args, 'audio_projector_init_scale', None)
        self.config.audio_projector_hidden_size = getattr(model_args, 'audio_projector_hidden_size', None)
        self.config.audio_projector_expand = getattr(model_args, 'audio_projector_expand', None)
        if audio_tower is not None:
            if self.get_audio_tower() is None:
                self.audio_tower = build_audio_tower(model_args, delay_load=False)
            else:
                if fsdp is not None and len(fsdp) > 0 and not deepspeed and isinstance(self.audio_tower, list):
                    self.audio_tower = self.audio_tower[0]
                self.audio_tower.load_model()
            self.config.mm_audio_hidden_size = self.audio_tower.hidden_size

            # Always (re)build the audio compressor so that a changed projector type / mode takes effect.
            self.mm_audio_projector = build_mm_projector(self.config, projector_type=self.config.mm_audio_projector_type)
            print(f"mm_audio_projector parameter size: {sum(p.numel() for p in self.mm_audio_projector.parameters())}")
            for p in self.mm_audio_projector.parameters():
                p.requires_grad = True

            if pretrain_mm_mlp_adapter is not None:
                audio_w = get_w(mm_projector_weights, 'mm_audio_projector')
                if len(audio_w) > 0:
                    print(f"Load pretrained mm_audio_projector weights from {pretrain_mm_mlp_adapter}")
                    self.mm_audio_projector.load_state_dict(audio_w, strict=not model_args.ignore_mismatched_sizes)
                else:
                    print(f"[warning] no mm_audio_projector weights found in {pretrain_mm_mlp_adapter}; audio compressor is randomly initialized")


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def set_mm_projector(self, mm_projector):
        return self.get_model().set_mm_projector(mm_projector)

    def get_audio_tower(self):
        return self.get_model().get_audio_tower()

    def get_mm_audio_projector(self):
        return self.get_model().get_mm_audio_projector()

    def set_mm_audio_projector(self, mm_audio_projector):
        return self.get_model().set_mm_audio_projector(mm_audio_projector)

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_audios(self, audio_values, audio_masks):
        """Encode padded log-Mel windows into compressed audio tokens.

        Args:
            audio_values: list (per sample) of tensors (n_windows, n_mels, 3000) or a stacked tensor
                (batch, n_windows, n_mels, 3000). Each window is a 30 s log-Mel spectrogram.
            audio_masks: matching (n_windows, 3000) frame-level attention masks (1=valid, 0=pad).

        Returns:
            list of tensors, one per sample, of shape (num_audio_tokens, hidden_size).
        """
        if isinstance(audio_values, list):
            audio_values = [x.unsqueeze(0) if x.ndim == 2 else x for x in audio_values]
        if isinstance(audio_masks, list):
            audio_masks = [x.unsqueeze(0) if x.ndim == 1 else x for x in audio_masks]
        audio_split_sizes = [audio_value.shape[0] for audio_value in audio_values]
        concat_audio_values = torch.cat([audio_value for audio_value in audio_values], dim=0)
        concat_audio_masks = torch.cat([audio_mask for audio_mask in audio_masks], dim=0)

        batch_size, mel_bins, max_mel_seq_len = concat_audio_values.size()
        assert max_mel_seq_len == 3000, f"max_mel_seq_len should be 3000 (30 s windows), but got {max_mel_seq_len}"

        audio_tower = self.get_model().get_audio_tower()
        audio_projector = self.get_model().get_mm_audio_projector()

        # Build the encoder attention mask (Qwen2-Audio style): 3000 mel frames -> 1500 conv positions.
        feat_lengths, output_lengths = audio_tower.model._get_feat_extract_output_lengths(concat_audio_masks.sum(-1))
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        seq_range = (
            torch.arange(0, max_seq_len, dtype=feat_lengths.dtype, device=feat_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        padding_mask = seq_range >= lengths_expand  # True where pad

        attn_mask = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len, max_seq_len)
        attn_mask = attn_mask.to(dtype=audio_tower.model.conv1.weight.dtype, device=audio_tower.model.conv1.weight.device)
        attn_mask[attn_mask.bool()] = float("-inf")

        with torch.no_grad():
            audio_out = audio_tower(
                concat_audio_values.to(device=audio_tower.device, dtype=audio_tower.dtype), attention_mask=attn_mask
            )
        raw_audio_features = audio_out.last_hidden_state  # (B, 750, D) at 25 Hz
        max_audio_tokens = raw_audio_features.size(1)

        # Group the 30 s windows back per sample (dropping padded positions) and compress.
        audio_features = []
        cursor = 0
        for sample_chunk_count in audio_split_sizes:
            sample_chunks = []
            for i in range(sample_chunk_count):
                idx = cursor + i
                valid_len = min(int(output_lengths[idx].item()), max_audio_tokens)
                sample_chunks.append(raw_audio_features[idx, :valid_len])
            cursor += sample_chunk_count
            sample_feat = torch.cat(sample_chunks, dim=0) if len(sample_chunks) > 1 else sample_chunks[0]  # (T, D)

            if isinstance(audio_projector, AudioCompressor):
                # Periodic-query compressor: reshape (T, D) -> (ceil(T/R), R, D), zero-padding the tail.
                block = audio_projector.config.block_size
                t_len, hidden_dim = sample_feat.shape
                if t_len % block != 0:
                    sample_feat = torch.cat([sample_feat, sample_feat.new_zeros(block - (t_len % block), hidden_dim)], dim=0)
                sample_feat = audio_projector(sample_feat.view(-1, block, hidden_dim))
            elif "patchwise_mlp" in self.config.mm_audio_projector_type:
                # Avg Pool baseline: MLP, then average every R tokens (tail block averaged as well).
                block_match = re.search(r'audio_(\d+)patchwise_mlp', self.config.mm_audio_projector_type)
                block_size = int(block_match.group(1))
                sample_feat = audio_projector(sample_feat)
                t_len = sample_feat.shape[0]
                num_complete_blocks = t_len // block_size
                remainder = t_len % block_size
                parts = []
                if num_complete_blocks > 0:
                    parts.append(sample_feat[: num_complete_blocks * block_size].view(num_complete_blocks, block_size, -1).mean(dim=1))
                if remainder > 0:
                    parts.append(sample_feat[num_complete_blocks * block_size :].mean(dim=0, keepdim=True))
                sample_feat = torch.cat(parts, dim=0)
            else:
                # Plain MLP (no compression)
                sample_feat = audio_projector(sample_feat)

            audio_features.append(sample_feat)
        return audio_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        audio_values=None,
        audio_masks=None,
    ):

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')

        if type(images) is list or images.ndim == 5:  # video: (batch, num_frames, 3, H, W)
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)  # (sum_of_frames, 3, H, W)
            split_sizes = [image.shape[0] for image in images]  # frames per sample

            # Long clips are encoded in five chunks to bound peak activation memory.
            if (
                "clip-vit-large-patch14-336" in self.config.mm_vision_tower
                or "siglip-so400m" in self.config.mm_vision_tower
                or "siglip2-so400m" in self.config.mm_vision_tower
            ) and len(concat_images) >= 20:
                with torch.no_grad():
                    fifth_idx = len(concat_images) // 5
                    image_features_a = self.get_model().get_vision_tower()(concat_images[:fifth_idx])
                    image_features_b = self.get_model().get_vision_tower()(concat_images[fifth_idx : 2 * fifth_idx])
                    image_features_c = self.get_model().get_vision_tower()(concat_images[2 * fifth_idx : 3 * fifth_idx])
                    image_features_d = self.get_model().get_vision_tower()(concat_images[3 * fifth_idx : 4 * fifth_idx])
                    image_features_e = self.get_model().get_vision_tower()(concat_images[4 * fifth_idx :])
                    image_features = torch.cat(
                        (image_features_a, image_features_b, image_features_c, image_features_d, image_features_e),
                        dim=0,
                    )
            else:
                image_features = self.get_model().get_vision_tower()(
                    concat_images
                )

            image_features = self.get_model().mm_projector(image_features)
            image_features = torch.split(image_features, split_sizes, dim=0)

            # ------------------------------------------------------------------
            # Audio: encode + compress the soundtrack of each video sample
            # ------------------------------------------------------------------
            audio_features = None
            has_audio = audio_values is not None and self.get_model().get_audio_tower() is not None and (
                (isinstance(audio_values, torch.Tensor) and audio_values.numel() > 0)
                or (isinstance(audio_values, list) and len([v for v in audio_values if v is not None]) > 0)
            )
            if has_audio:
                assert audio_masks is not None, "audio_masks must be provided when audio_values is not None."
                audio_features = self.encode_audios(audio_values, audio_masks)

            if mm_patch_merge_type == 'flat_video_frame_with_end_token':
                processed_image_features = []
                for x in image_features:
                    newline_expanded = (
                        self.model.image_newframe.unsqueeze(0).unsqueeze(1).expand(x.size(0), 1, -1).to(x.device)
                    )
                    combined = torch.cat([x, newline_expanded], dim=1)
                    flattened = combined.flatten(0, 1)
                    processed_image_features.append(flattened)
                image_features = processed_image_features
            elif 'flat_video_frame_with_end_token_and_avepool' in mm_patch_merge_type:
                # ``..._avepool{S}``        : average-pool each frame by stride S; audio tokens (if any) are appended
                #                             after all visual tokens (non-interleaving, [V; A]).
                # ``..._avepool{S}_zigzag`` : time-aligned interleaving; the compressed audio tokens are split evenly
                #                             across frames and placed right after the visual tokens of each frame.
                stride_match = re.match(r'^flat_video_frame_with_end_token_and_avepool(\d+)(?:_zigzag)?$', mm_patch_merge_type)

                assert stride_match is not None, f"Unexpected mm_patch_merge_type: {mm_patch_merge_type}"

                ave_stride = int(stride_match.group(1))

                new_image_features = []
                for image_feature in image_features:

                    num_frames, num_tokens, num_dim = image_feature.shape
                    edge_tok_num = int(num_tokens**0.5)
                    image_feature = image_feature.view(
                        num_frames, edge_tok_num, edge_tok_num, -1
                    )
                    image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
                    image_feature = nn.functional.avg_pool2d(image_feature, ave_stride)
                    # scaled_shape = [math.ceil(edge_tok_num / interpolation_stride), math.ceil(edge_tok_num / interpolation_stride)]
                    image_feature = image_feature.permute(0, 2, 3, 1)
                    image_feature = image_feature.view(num_frames, -1, num_dim)
                    new_image_features.append(image_feature)

                interleave_audio = "zigzag" in mm_patch_merge_type
                video_counter = 0
                processed_image_features = []
                for x in new_image_features:
                    num_frames = x.size(0)
                    is_video = num_frames > 1
                    if is_video and audio_features is not None and interleave_audio:
                        # Time-aligned interleaving: distribute audio tokens evenly over frames
                        # (pad by repeating the last token so that every frame gets the same count).
                        this_audio_features = audio_features[video_counter].to(x.device, x.dtype)  # [M, D]
                        M = this_audio_features.size(0)
                        if M > 0:
                            D = this_audio_features.size(1)
                            tokens_per_frame = (M + num_frames - 1) // num_frames  # ceil(M / num_frames)
                            total_needed = tokens_per_frame * num_frames
                            if M < total_needed:
                                last_tok = this_audio_features[-1:].expand(total_needed - M, D)
                                this_audio_features = torch.cat([this_audio_features, last_tok], dim=0)
                            per_frame_audio = this_audio_features.view(num_frames, tokens_per_frame, D)
                            x = torch.cat([x, per_frame_audio], dim=1)  # [num_frames, V + tokens_per_frame, D]

                    newline_expanded = (
                        self.model.image_newframe.unsqueeze(0).unsqueeze(1).expand(x.size(0), 1, -1).to(x.device)
                    )
                    combined = torch.cat([x, newline_expanded], dim=1)
                    flattened = combined.flatten(0, 1)

                    if is_video and audio_features is not None and not interleave_audio:
                        # Non-interleaving: append all audio tokens after the visual tokens ([V; A]).
                        a_feat = audio_features[video_counter].to(flattened.device, flattened.dtype)
                        if a_feat.dim() == 1:
                            a_feat = a_feat.unsqueeze(0)
                        flattened = torch.cat([flattened, a_feat], dim=0)

                    if is_video:
                        video_counter += 1
                    processed_image_features.append(flattened)
                image_features = processed_image_features
            elif mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx][0]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
                # # multimodal LLM, but the current sample is not multimodal
                # # FIXME: this is a hacky fix, for deepspeed zero3 to work
                # half_len = cur_input_ids.shape[0] // 2
                # cur_image_features = image_features[cur_image_idx]
                # cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                # cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                # cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                # new_input_embeds.append(cur_input_embeds)
                # if labels is not None:
                #     new_labels.append(labels[batch_idx])
                # cur_image_idx += 1
                # continue
            image_token_indices = (
                [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            # len(cur_input_embeds_no_im)
            # 2
            # cur_input_embeds_no_im[0].size()
            # cur_input_embeds_no_im[1].size()
            cur_new_input_embeds = []
            cur_new_labels = []

            if num_images > 1:
                print("[WARNING] num_images > 1 observed, cast it to 1")
                num_images = 1

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    assert i == 0
                    # if image_token_indexes is not None and text_token_indexes is not None:
                    #     image_token_indexes[batch_idx] += len(cur_input_embeds_no_im[i]) + self.get_vision_tower().num_patches_per_side**2
                    #     text_token_indexes[batch_idx] += len(cur_input_embeds_no_im[i]) + cur_image_features.shape[0]
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, "<vpatch>", "<vrow_sep>", "<vframe_sep>"],
                special_tokens=True,
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            assert (
                not model_args.tune_mm_mlp_adapter
            ), "model_args.mm_use_im_start_end requires caution with frozen parameters when adding new tokens."

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
