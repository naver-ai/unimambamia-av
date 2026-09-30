# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Modified from https://github.com/naver-ai/mambamia/blob/main/MambaMia/mambamia_vid.py,
# which is inspired by and modified from
# https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/v0.3.0/lmms_eval/models/llava_vid.py

"""lmms-eval model wrapper for UniMambaMia-AV.

Register name: ``unimambamia_vid``. Set ``use_audio=True`` to feed the soundtrack and
``use_audio=False`` to evaluate the same checkpoint on muted video.

    accelerate launch --num_processes 8 -m lmms_eval --model unimambamia_vid \\
        --model_args pretrained=gwkrsrch/UniMambaMia-AV-Qwen2-7B,use_audio=True,max_frames_num=32,\\
video_fps=1.0,conv_template=qwen_2,attn_implementation=flash_attention_2 \\
        --tasks av_speakerbench_audiovisual --batch_size 1 --log_samples --output_path ./eval_logs

Videos are decoded in a background pool while the GPU is busy with the previous sample, which keeps
decoding (the bottleneck for hour-long clips with audio) off the critical path.
"""

import copy
import multiprocessing
import os
import queue
import threading
from collections import OrderedDict
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_qwen import LlavaQwenConfig
from llava.video_utils import sample_video_audio

AutoConfig.register("llava_qwen", LlavaQwenConfig)

CPU_COUNT = os.cpu_count()


def load_video_worker(video_path, max_frames_num, fps, rank=0, world_size=1, use_audio=False):
    """Decode one video in a worker process: sampled frames and, optionally, 30 s audio chunks."""
    start_idx = min(rank * (CPU_COUNT // world_size), CPU_COUNT - 1)
    end_idx = min((rank + 1) * (CPU_COUNT // world_size), CPU_COUNT)
    frames, audio_chunks, actual_fps, video_time = sample_video_audio(
        FPS=fps,
        video_file_path=video_path,
        MAX_Frame=max_frames_num,
        cpu_idx=np.random.randint(start_idx, end_idx),
        use_audio=use_audio,
        sample_rate=16000,
        debug=False,
    )
    return frames, actual_fps, video_time, audio_chunks


class LRUCacheDict(OrderedDict):
    """Keeps at most ``max_size`` entries, evicting the least recently used one."""

    def __init__(self, max_size=8, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.max_size:
            super().__delitem__(next(iter(self)))


class DocVisualAsyncLoader:
    """Decodes the videos of ``requests`` ahead of the model, in a bounded queue."""

    def __init__(self, requests, task_dict, max_frames_num, fps, rank, world_size, buffer_size=16, use_audio=False):
        self.requests = requests
        self.task_dict = task_dict
        self.max_frames_num = max_frames_num
        self.fps = fps
        self.rank = rank
        self.world_size = world_size
        self.use_audio = use_audio

        self.cache = LRUCacheDict(max_size=8)  # several benchmarks ask multiple questions per video
        self.q = queue.Queue(maxsize=buffer_size)
        self.pool = multiprocessing.Pool(processes=4)

        self.thread = threading.Thread(target=self._producer, daemon=True)
        self.thread.start()

    def _producer(self):
        for i, reg in enumerate(self.requests):
            _, _, doc_to_visual, doc_id, task, split = reg.args
            try:
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])
                if not visuals:
                    raise RuntimeError("doc_to_visual returned no video")
                video_path = visuals[0]

                if video_path in self.cache:
                    future = self.cache[video_path]
                else:
                    future = self.pool.apply_async(
                        load_video_worker,
                        (video_path, self.max_frames_num, self.fps, self.rank, self.world_size, self.use_audio),
                    )
                    self.cache[video_path] = future
                self.q.put((i, video_path, future))
            except Exception as e:
                eval_logger.error(f"Error loading video for doc_id={doc_id}: {e}")
                self.q.put((i, None, None))
        self.q.put(None)

    def get_next(self):
        """Return ``(index, video_path, frames, fps, duration, audio_chunks)``, or None when done."""
        item = self.q.get()
        if item is None:
            return None
        i, video_path, future = item
        try:
            frames, actual_fps, video_time, audio_chunks = future.get()
            return i, video_path, frames, actual_fps, video_time, audio_chunks
        except Exception as e:
            eval_logger.error(f"Error decoding {video_path}: {e}")
            return i, video_path, None, None, None, None

    def join(self):
        self.thread.join()
        self.pool.close()
        self.pool.join()


def expand2square(frame, background_color):
    """Pad an ``(H, W, C)`` uint8 frame to a square with ``background_color``."""
    h, w, c = frame.shape
    if w == h:
        return frame
    size = max(h, w)
    result = np.full((size, size, c), background_color, dtype=np.uint8)
    if w > h:
        top = (w - h) // 2
        result[top : top + h, :, :] = frame
    else:
        left = (h - w) // 2
        result[:, left : left + w, :] = frame
    return result


@register_model("unimambamia_vid")
class UniMambaMiaVid(lmms):
    """Audio-visual Video-LLM with a Mamba-based audio compressor."""

    def __init__(
        self,
        pretrained: str = "gwkrsrch/UniMambaMia-AV-Qwen2-7B",
        use_audio: Union[bool, str] = True,
        max_frames_num: int = 32,
        video_fps: float = 1.0,
        conv_template: str = "qwen_2",
        add_time_instruction: Union[bool, str] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "float16",
        model_max_length: Optional[int] = 50000,
        device: str = "cuda:0",
        device_map: str = "cuda:0",
        batch_size: Union[int, str] = 1,
        use_cache: bool = True,
        truncation: bool = True,
        tie_weights: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> None:
        """
        Args:
            pretrained: HF hub id or local path of the checkpoint.
            use_audio: feed the soundtrack (``True``) or evaluate the muted video (``False``).
            max_frames_num, video_fps: visual frame budget; the released model was trained with 32 @ 1.0.
            conv_template: conversation template (``qwen_2`` for the Qwen2-7B backbone).
            add_time_instruction: prepend "The video lasts for ..."; the released model was trained without it.
        """
        super().__init__()
        if kwargs:
            eval_logger.warning(f"Ignoring unexpected model_args: {sorted(kwargs)}")

        def _as_bool(value):
            return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")

        accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(weeks=52))])
        if accelerator.num_processes > 1 or device_map not in ("auto", "balanced_low_0"):
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        self.pretrained = pretrained
        self.use_audio = _as_bool(use_audio)
        self.add_time_instruction = _as_bool(add_time_instruction)
        self.model_name = get_model_name_from_path(pretrained)
        self.max_frames_num = int(max_frames_num)
        self.fps = float(video_fps)
        self.conv_template = conv_template
        self.torch_dtype = torch_dtype
        self.temperature = temperature
        self.top_p = top_p
        self.use_cache = use_cache
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)

        self._tokenizer, self._model, self._image_processor, self._audio_processor, self._max_length = load_pretrained_model(
            model_path=pretrained,
            device=self._device,
            device_map=self.device_map,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )

        if self.use_audio and self._audio_processor is None:
            raise ValueError("use_audio=True but this checkpoint has no audio tower (config.mm_audio_tower is unset).")
        if not self.use_audio:
            self._audio_processor = None  # muted evaluation: never read the soundtrack

        if model_max_length is not None:
            self._tokenizer.model_max_length = model_max_length
            self._model.config.tokenizer_model_max_length = model_max_length
            self._model.model.config.tokenizer_model_max_length = model_max_length

        if self._tokenizer.pad_token_id is None and "qwen" in self._tokenizer.name_or_path.lower():
            self._tokenizer.pad_token_id = 151643  # <|endoftext|>

        self._config = self._model.config
        self._model.eval()
        if tie_weights and self._model.config.tie_word_embeddings:
            self._model.tie_weights()

        eval_logger.info(f"use_audio={self.use_audio}, max_frames={self.max_frames_num}, fps={self.fps}, template={self.conv_template}")

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True,
                    train_micro_batch_size_per_gpu=self.batch_size_per_gpu,
                    train_batch_size=self.batch_size_per_gpu * accelerator.num_processes,
                )
            if accelerator.distributed_type in (DistributedType.FSDP, DistributedType.DEEPSPEED):
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._model.to(self._device)
            self._rank = 0
            self._world_size = 1

    # ------------------------------------------------------------------ lmms API
    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        encoding = self.tokenizer.encode(string, add_special_tokens=bool(add_special_tokens))
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for UniMambaMiaVid")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for UniMambaMiaVid")

    # ------------------------------------------------------------------ inference
    def _prepare_frames(self, frames):
        """``(num_frames, H, W, 3)`` uint8 -> ``(num_frames, 3, h, w)`` on the model device."""
        if self.model.config.image_aspect_ratio == "pad":
            pad_color = tuple(int(x * 255) for x in self._image_processor.image_mean)
            frames = [expand2square(frame, pad_color) for frame in frames]
        processed = [self._image_processor.preprocess(frame, return_tensors="pt")["pixel_values"][0] for frame in frames]
        video = torch.stack(processed, dim=0).to(self._device)
        return video.bfloat16() if self.torch_dtype == "bfloat16" else video.half()

    def _prepare_audio(self, audio_chunks):
        """30 s waveform chunks -> ``(num_windows, 128, 3000)`` log-Mel features and their masks."""
        features = self._audio_processor(
            audio_chunks,
            sampling_rate=self._audio_processor.sampling_rate,
            return_attention_mask=True,
            padding="max_length",
        )
        return torch.from_numpy(features.input_features), torch.from_numpy(features.attention_mask)

    def generate_until(self, requests) -> List[str]:
        loader = DocVisualAsyncLoader(
            requests=requests,
            task_dict=self.task_dict,
            max_frames_num=self.max_frames_num,
            fps=self.fps,
            rank=self.rank,
            world_size=self.world_size,
            use_audio=self.use_audio,
        )

        res = []
        pbar = tqdm(total=len(requests), desc=f"Model responding at rank {self.rank}")
        try:
            for idx, reg in enumerate(requests):
                item = loader.get_next()
                if item is None:
                    break
                i, video_path, frames, actual_fps, video_time, audio_chunks = item
                assert i == idx, f"Producer/consumer index mismatch: {i} != {idx}"

                contexts, gen_kwargs, _, _, _, _ = reg.args
                audio_values, audio_masks = [], []

                try:
                    if frames is None or len(frames) == 0:
                        raise RuntimeError("no frames decoded")
                    videos = [self._prepare_frames(frames)]

                    if self._audio_processor is not None:
                        if audio_chunks:
                            values, masks = self._prepare_audio(audio_chunks)
                            audio_values.append(values)
                            audio_masks.append(masks)
                        else:
                            eval_logger.warning(f"No audio track in {video_path}; answering from video only.")
                except Exception as e:
                    eval_logger.info(f"Video {video_path} can not load ({e}), check the source")
                    res.append(f"Video {video_path} can not load, check the source")
                    pbar.update(1)
                    continue

                prompt_text = contexts
                if self.add_time_instruction:
                    prompt_text = (
                        f"The video lasts for {video_time:.3f} seconds, and {len(frames)} frames are sampled from it "
                        f"with FPS {actual_fps:.3f}. Please answer the following questions related to this video.\n{prompt_text}"
                    )
                if self.model.config.mm_use_im_start_end:
                    prompt_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
                else:
                    prompt_text = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

                conv = copy.deepcopy(conv_templates[self.conv_template])
                conv.append_message(conv.roles[0], prompt_text)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self._device)

                pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO and conv.sep else conv.sep2
                stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

                gen_kwargs.setdefault("max_new_tokens", 1024)
                gen_kwargs.setdefault("temperature", 1.0)
                gen_kwargs.setdefault("top_p", None)
                gen_kwargs.setdefault("num_beams", 1)
                gen_kwargs.setdefault("repetition_penalty", 1.0)

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        inputs=input_ids,
                        images=videos,
                        audio_values=audio_values or None,
                        audio_masks=audio_masks or None,
                        pad_token_id=pad_token_id,
                        use_cache=self.use_cache,
                        stopping_criteria=[stopping_criteria],
                        do_sample=False,  # greedy decoding throughout the paper
                        temperature=self.temperature if self.temperature is not None else gen_kwargs["temperature"],
                        top_p=self.top_p if self.top_p is not None else gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        repetition_penalty=gen_kwargs["repetition_penalty"],
                    )

                outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                eval_logger.debug(f"Question: {prompt_text}\nAnswer: {outputs}")
                res.append(outputs)
                pbar.update(1)
        finally:
            loader.join()
            pbar.close()

        return res
