# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Modified from https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/train/train.py
# and https://github.com/naver-ai/mambamia/blob/main/MambaMia/train.py
# Below is the original copyright from LLaVA:
#
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import numpy as np
import torch
from math import ceil

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.video_utils import sample_video_audio, malloc_trim, _drop_file_cache

from pprint import pprint
import ast
from PIL import Image
from io import BytesIO
import gc

import pickle
import io

local_rank = None

CPU_COUNT = os.cpu_count()
print(f'Number of CPUs: {CPU_COUNT}')

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="qwen_2")
    freeze_backbone: bool = field(default=False)
    full_tuning: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    audio_tower: Optional[str] = field(default=None)  # e.g. gwkrsrch2/qwen2-audio-encoder-from-qwen2-audio-7b-instruct
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_audio_projector_type: Optional[str] = field(default='linear')  # e.g. audio_projector_mambamia2uni
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    image_grid_pinpoints: Optional[str] = field(default='[(1,1)]')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # Audio compressor (UniMambaMia): mode encodes the compression ratio R as "{R}patchwise"
    audio_projector_mode: Optional[str] = field(default='per_from1to1frames_25patchwise_25tokperframe')
    audio_projector_num_layers: Optional[int] = field(default=2)
    audio_projector_init_scale: Optional[float] = field(default=None)  # gated-attention init scale (default 1e-3)
    audio_projector_hidden_size: Optional[int] = field(default=None)  # default 3072
    audio_projector_expand: Optional[float] = field(default=None)  # default 2.0


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    multiple: str = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    video_fps: float = 1.0
    max_frame: int = 10
    use_audio: bool = False
    sample_rate: int = 16000
    max_video_duration: Optional[float] = field(default=900.0, metadata={"help": "Skip videos longer than this (seconds) when use_audio=True to bound audio memory. None = no limit."})
    add_time_instruction: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_vision_encoder: bool = field(default=False)
    freeze_two_encoder: bool = field(default=False)  # freeze both the vision and the audio encoder
    torch_compile: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)
    delay_load: bool = field(default=True)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers.modeling_utils import is_fsdp_enabled
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        import gc

        if is_fsdp_enabled():
            print("FSDP enabled")
            with FSDP.state_dict_type(
                trainer.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=False, rank0_only=True),
            ):
                full_sd = trainer.model.state_dict()

            weight_to_save = {
                k: v for k, v in full_sd.items() if "mm_projector" in k or "mm_audio_projector" in k
            }

            gc.collect()
        else:
            print("FSDP disabled")
            # Only save Adapter
            keys_to_match = ['mm_projector', 'mm_audio_projector']
            if getattr(trainer.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        trainer.model.config.save_pretrained(output_dir)

        return

    # if trainer.deepspeed:
    torch.cuda.synchronize()
    trainer.save_model(output_dir)
    trainer.model.config.save_pretrained(output_dir)
    return

    # state_dict = trainer.model.state_dict()
    # if trainer.args.should_save:
    #     cpu_state_dict = {
    #         key: value.cpu()
    #         for key, value in state_dict.items()
    #     }
    #     del state_dict
    #     trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens

    # # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    # tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

def preprocess_qwen_2_5(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens

    # # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    # tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    version = conversation_lib.default_conversation.version
    if version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if version == "qwen_2_5":
        return preprocess_qwen_2_5(sources, tokenizer, has_image=has_image)
    raise ValueError(f"Unsupported conversation version: {version}. Use --version qwen_2 or qwen_2_5.")


def expand2square_np(np_img, background_color):
    """
    Expand image to square by padding.
    
    Args:
        np_img: numpy array with shape (H, W, C)
        background_color: tuple of (R, G, B) values (0-255)
    """
    h, w, c = np_img.shape
    if w == h:
        return np_img
    elif w > h:
        diff = w - h
        top_pad = diff // 2
        bottom_pad = diff - top_pad
        result = np.full((w, w, c), background_color, dtype=np.uint8)
        result[top_pad:top_pad + h, 0:w, :] = np_img
        return result
    else:
        diff = h - w
        left_pad = diff // 2
        right_pad = diff - left_pad
        result = np.full((h, h, c), background_color, dtype=np.uint8)
        result[0:h, left_pad:left_pad + w, :] = np_img
        return result
    
def expand2square_pil(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.cumulative_dataset_lengths = []
        data_path_list = ast.literal_eval(data_path)
        # pprint(data_path_list)

        if data_args.multiple is not None:
            multiple = ast.literal_eval(data_args.multiple)
            print("multiple: ")
            pprint(multiple)
            assert len(data_path_list) == len(multiple)

        self.sample_index = []  # [(file_idx, line_idx), ...]
        self.jsonl_paths = []

        for data_path_idx, data_path in enumerate(data_path_list):
            multi = multiple[data_path_idx] if data_args.multiple is not None else 1

            self.jsonl_paths.append(data_path)
            assert data_path_idx == len(self.jsonl_paths)-1

            with open(data_path, "r", encoding="utf-8") as f:
                num_samples = sum(1 for _ in f)

            for _ in range(multi):
                for i in range(num_samples):
                    self.sample_index.append((len(self.jsonl_paths)-1, i))
                self.cumulative_dataset_lengths.append(len(self.sample_index))
        self._length = len(self.sample_index)

        self.line_offsets = []
        for jsonl_path in self.jsonl_paths:
            offsets = []
            offset = 0
            with open(jsonl_path, "rb") as f:
                for line in f:
                    offsets.append(offset)
                    offset += len(line)
            self.line_offsets.append(offsets)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

        self.debug_flag = True

        self.image_data_source = list()
        image_folder_path_list = ast.literal_eval(self.data_args.image_folder)
        # pprint(image_folder_path_list)
        assert len(data_path_list) == len(image_folder_path_list)

        for dirpath_idx, dirpath in enumerate(image_folder_path_list):
            multi = multiple[dirpath_idx] if data_args.multiple is not None else 1
            for _ in range(multi):
                self.image_data_source.append(dirpath)

    def __len__(self):
        return self._length
    
    def _get_json_dict(self, file_idx, line_idx):
        path = self.jsonl_paths[file_idx]
        with open(path, "r", encoding="utf-8") as f:
            f.seek(self.line_offsets[file_idx][line_idx])
            line = f.readline()
        return json.loads(line)

    def get_image_data_source_from_idx(self, idx):
        for dataset_idx, cumulative_dataset_length in enumerate(self.cumulative_dataset_lengths):
            if dataset_idx == 0:
                if idx < cumulative_dataset_length:
                    return self.image_data_source[dataset_idx]
            else:
                if self.cumulative_dataset_lengths[dataset_idx - 1] <= idx and idx < cumulative_dataset_length:
                    return self.image_data_source[dataset_idx]
        print("[ERROR] get_image_data_source_from_idx")
        dataset_idx = -1
        return self.image_data_source[dataset_idx]

    @property
    def lengths(self):
        length_list = []
        for file_idx, line_idx in self.sample_index:
            sample = self._get_json_dict(file_idx, line_idx)
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for file_idx, line_idx in self.sample_index:
            sample = self._get_json_dict(file_idx, line_idx)
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i, retry=10) -> Dict[str, torch.Tensor]:
        if retry <= 0:
            raise RuntimeError("Retry limit 10 exceeded")
        
        source = None
        try:
            file_idx, line_idx = self.sample_index[i]
            source = self._get_json_dict(file_idx, line_idx)
            has_image = 'image' in source or 'video' in source

            processor = self.data_args.image_processor
            audio_processor = getattr(self.data_args, "audio_processor", None) if self.data_args.use_audio else None
            if self.data_args.use_audio and audio_processor is None:
                raise ValueError("data_args.use_audio is True but no audio_processor is set (is --audio_tower given?)")
            if audio_processor is not None and audio_processor.sampling_rate != self.data_args.sample_rate:
                raise ValueError(f"audio_processor.sampling_rate({audio_processor.sampling_rate}) != data_args.sample_rate({self.data_args.sample_rate})")
            audio_value = None
            audio_mask = None
            audio_chunks = None

            if 'video' in source:
                video_file = source['video']

                image_source = self.get_image_data_source_from_idx(i)
                image, audio_chunks, actual_FPS, video_length = sample_video_audio(
                    FPS=self.data_args.video_fps,
                    video_file_path=os.path.join(image_source, video_file),
                    MAX_Frame=self.data_args.max_frame,
                    cpu_idx=i // CPU_COUNT,
                    use_audio=self.data_args.use_audio,
                    sample_rate=self.data_args.sample_rate,
                    debug=False,
                    max_video_duration=self.data_args.max_video_duration if self.data_args.use_audio else None,
                )

                if self.data_args.image_aspect_ratio == 'pad':
                    pad_color = tuple(int(x * 255) for x in processor.image_mean)
                    image = [
                        processor.preprocess(
                            expand2square_np(frame, pad_color), return_tensors='pt'
                        )['pixel_values'][0] for frame in image
                    ]

                else:
                    image = [processor.preprocess(frame, return_tensors='pt')['pixel_values'][0]
                            for frame in image]

                image = torch.stack(image, dim=0)

                if audio_processor is not None:
                    if not isinstance(audio_chunks, list) or len(audio_chunks) == 0:
                        raise ValueError("use_audio=True but no audio chunks were returned (LMDB videos do not carry audio; use raw video files)")
                    preprocess_results = audio_processor(
                        audio_chunks,
                        sampling_rate=audio_processor.sampling_rate,
                        return_attention_mask=True,
                        padding="max_length",
                    )  # input_features: (n_windows, 128, 3000), attention_mask: (n_windows, 3000)
                    # .copy() breaks the reference chain to WhisperFeatureExtractor's internal torch storage
                    audio_value = torch.tensor(preprocess_results.input_features.copy())
                    audio_mask = torch.tensor(preprocess_results.attention_mask.copy())
                    del preprocess_results
                    del audio_chunks
                    audio_chunks = None
                    gc.collect()

                if self.data_args.add_time_instruction:
                    # time_instruciton = f"The video lasts for {video_length:.2f} seconds, and {len(image)} frames are sampled from it with FPS {actual_FPS:.2f}. Please answer the following questions related to this video."
                    time_instruciton = f"The video lasts for {video_length:.3f} seconds, and {len(image)} frames are sampled from it with FPS {actual_FPS:.3f}. Please answer the following questions related to this video."
                    source["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{source["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'

                source = preprocess_multimodal(
                    [source["conversations"]],
                    self.data_args)[0]
            elif 'image' in source:
                image_file = source['image']

                image_source = self.get_image_data_source_from_idx(i)
                if isinstance(image_source, str):
                    image = Image.open(os.path.join(image_source, image_file)).convert('RGB')
                else:
                    image = image_source.get(image_file.encode("utf-8"))
                    try:
                        image = Image.open(BytesIO(image)).convert("RGB")
                    except:
                        print(f"[warning] image error with {image_file}")
                        gc.collect()
                        return self.__getitem__(np.random.randint(len(self.sample_index)), retry=retry-1)

                if self.data_args.image_aspect_ratio == 'pad':
                    image = expand2square_pil(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                source = preprocess_multimodal(
                    [source["conversations"]],
                    self.data_args)[0]
            else:
                source = source["conversations"]
            data_dict = preprocess(
                [source],
                self.tokenizer,
                has_image=has_image)
            
            del source
            source = None

            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0][:self.tokenizer.model_max_length],
                                labels=data_dict["labels"][0][:self.tokenizer.model_max_length])
            # image exist in the data
            if has_image:
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

            # audio (None for image-only / audio-less samples)
            data_dict["audio_value"] = audio_value
            data_dict["audio_mask"] = audio_mask

            # DataLoader worker memory hygiene: every sample when audio is on, otherwise every 10th
            if audio_processor is not None:
                gc.collect()
                malloc_trim()
            elif i % 10 == 0:
                gc.collect()
                malloc_trim()

            return data_dict
        
        except Exception as e:
            print(f"[warning] data index error at {i}: {e}")
            image = None
            audio_chunks = None
            audio_value = None
            audio_mask = None
            source = None
            gc.collect()
            malloc_trim()
            return self.__getitem__(np.random.randint(len(self.sample_index)), retry=retry-1)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 151643

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if isinstance(images[0], list):
                images = torch.stack([torch.stack(img, dim=0) for img in images], dim = 0)
                batch['images'] = images
            else:
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch['images'] = torch.stack(images)
                else:
                    batch['images'] = images
                    
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'audio_value' in instances[0]:
            audio_values = [instance['audio_value'] for instance in instances]
            if all(x is not None and x.shape == audio_values[0].shape for x in audio_values):
                batch['audio_values'] = torch.stack(audio_values)
            else:
                batch['audio_values'] = audio_values

        if 'audio_mask' in instances[0]:
            audio_masks = [instance['audio_mask'] for instance in instances]
            if all(x is not None and x.shape == audio_masks[0].shape for x in audio_masks):
                batch['audio_masks'] = torch.stack(audio_masks)
            else:
                batch['audio_masks'] = audio_masks

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    from_pretrained_args = {"ignore_mismatched_sizes": training_args.ignore_mismatched_sizes}
    model_args.ignore_mismatched_sizes = training_args.ignore_mismatched_sizes

    print(f"########## training_args.delay_load: {training_args.delay_load}")

    if model_args.vision_tower is not None:

        if 'qwen' in model_args.model_name_or_path.lower():
            print(f'####### Loading LlavaQwenForCausalLM from {model_args.model_name_or_path}')
            model = LlavaQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                delay_load=training_args.delay_load,
                **from_pretrained_args
            )
        else:
            raise ValueError(
                f"Expected a Qwen2-based checkpoint, got {model_args.model_name_or_path}. "
                "This release ships the Qwen2 backbone only."
            )
    else:
        raise ValueError("--vision_tower is required")

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # Enable full tuning if requested
    if getattr(model_args, "full_tuning", False):
        print("Full tuning: enabling all model parameters to require gradients.")
        for param in model.parameters():
            param.requires_grad = True

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )

    if model_args.version not in conversation_lib.conv_templates:
        raise ValueError(f"Unknown conversation version: {model_args.version}. Use qwen_2 or qwen_2_5.")
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    rank0_print(f"[INFO] conversation template: {model_args.version}")

    if model_args.vision_tower is not None:
        print(f"training_args.fsdp_config : {training_args.fsdp_config }")
        print(f"training_args.deepspeed : {training_args.deepspeed }")
        print(f"model_args.vision_tower : {model_args.vision_tower }")

        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp_config,
            deepspeed=training_args.deepspeed
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        if model_args.audio_tower is not None:
            audio_tower = model.get_audio_tower()
            audio_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            data_args.audio_processor = audio_tower.audio_processor

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = model_args.image_grid_pinpoints
        data_args.image_grid_pinpoints = model_args.image_grid_pinpoints
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            print(f"######## tune_mm_mlp_adapter is set to True")
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            if model_args.audio_tower is not None:
                # module-only alignment: also train the audio compressor
                for p in model.get_model().get_mm_audio_projector().parameters():
                    p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if model.get_model().get_vision_tower().select_layer != model_args.mm_vision_select_layer:
        print(f"########## model.get_model().get_vision_tower().select_layer: {model.get_model().get_vision_tower().select_layer} vs model_args.mm_vision_select_layer: {model_args.mm_vision_select_layer}")
        model.get_model().get_vision_tower().select_layer = model_args.mm_vision_select_layer
    if hasattr(model, "get_vision_tower") and hasattr(model.get_vision_tower(), "remove_unused_layers"):
        model.get_vision_tower().remove_unused_layers()
    if training_args.freeze_vision_encoder:
        print("freeze_vision_encoder is set to True")
        for p in model.get_model().get_vision_tower().parameters():
            p.requires_grad = False

    if training_args.freeze_two_encoder:
        print("freeze_two_encoder is set to True: freezing the vision encoder and the audio encoder")
        for p in model.get_model().get_vision_tower().parameters():
            p.requires_grad = False
        if model.get_model().get_audio_tower() is not None:
            for p in model.get_model().get_audio_tower().parameters():
                p.requires_grad = False

    if training_args.torch_compile:
        print(f"Train with a torch2.0 compile (PyTorch {torch.__version__}).")
        model.compile()
        torch._dynamo.config.verbose = False
        torch._dynamo.config.suppress_errors = True

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if training_args.bf16:
        for name, module in model.named_modules():
            module = module.to(torch.bfloat16)

    print(f"training_args.deepspeed: {training_args.deepspeed}")
    print(f"training_args.fsdp_config and training_args.fp16: {training_args.fsdp_config} and {training_args.fp16}")
    if not training_args.deepspeed and training_args.fsdp_config and training_args.fp16:
        for name, module in model.named_modules():
            try:
                module.to(torch.float16)
            except Exception as e:
                pass
    if not training_args.bf16 and not training_args.fp16:
        print(f"######### Using FP32")
        for name, module in model.named_modules():
            try:
                module.to(torch.float32)
            except Exception as e:
                pass

    if training_args.learning_rate != 0.0:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    else:
        print("we are not training the model, just saving it.")
    
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()