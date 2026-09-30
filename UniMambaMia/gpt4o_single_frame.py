# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Adopted from lmms-eval from https://github.com/EvolvingLMMs-Lab/lmms-eval (lmms_eval/models/gpt4v.py). Below is the original copyright:
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
#
# GPT-4o probe used for the single-frame benchmark audit. With ``max_frames_num=1`` (default) only the
# temporally central frame of each video is sent, resized so that its longer side is at most 384 px;
# no audio and no other frames are provided. ``continual_mode`` caches responses per rank so that an
# interrupted run can be resumed. Register name: ``gpt4o``.
#
#   accelerate launch --num_processes 8 -m lmms_eval --model gpt4o \
#       --model_args model_version=gpt-4o-2024-08-06,max_frames_num=1,continual_mode=True,response_persistent_folder=./logs/gpt4o_1frame \
#       --tasks avqa_hard --log_samples --output_path ./eval_logs/gpt4o_single_frame   # add --gen_kwargs temperature=1.0 for the second run

import base64
import json
import os
import time
from copy import deepcopy
from io import BytesIO
from typing import List, Tuple

os.environ["DECORD_EOF_RETRY_MAX"] = "20480"

import numpy as np
import requests as url_requests
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from PIL import Image

API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 30
from loguru import logger as eval_logger

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }


@register_model("gpt4o")
class GPT4O(lmms):
    def __init__(
        self,
        model_version: str = "gpt-4o",
        max_frames_num: int = 1,
        max_image_size: int = 384,
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.max_frames_num = max_frames_num
        self.max_image_size = max_image_size
        self.image_token = "<image>"
        self.timeout = timeout

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

        # Continual mode: per-rank cache file to support multi-process
        self.continual_mode = continual_mode
        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")

            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder,
                f"{self.model_version}_rank{self._rank}_response.json",
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

    def resize_image(self, image: Image):
        """Resize so the longer side <= max_image_size. No upscaling."""
        w, h = image.size
        if max(w, h) > self.max_image_size:
            scale = self.max_image_size / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return image

    def encode_image(self, image: Image):
        image = self.resize_image(image)
        output_buffer = BytesIO()
        image.save(output_buffer, format="JPEG", quality=90)
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    def encode_video(self, video_path, for_get_frames_num):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)

        if for_get_frames_num == 1:
            # Extract the center frame
            frame_idx = [total_frame_num // 2]
        else:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()

        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = self.resize_image(Image.fromarray(frame))
            output_buffer = BytesIO()
            img.save(output_buffer, format="JPEG", quality=90)
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = f"{task}___{split}___{doc_id}"
                if doc_uuid in self.response_cache:
                    response_text = self.response_cache[doc_uuid]
                    if response_text:
                        res.append(response_text)
                        pbar.update(1)
                        continue

            try:
                visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                visuals = self.flatten(visuals)
            except Exception as e:
                eval_logger.error(f"Error loading visual for doc_id={doc_id}: {str(e)}")
                res.append("")
                pbar.update(1)
                if self.continual_mode is True:
                    doc_uuid = f"{task}___{split}___{doc_id}"
                    self.response_cache[doc_uuid] = ""
                    with open(self.response_persistent_file, "w") as f:
                        json.dump(self.response_cache, f)
                continue

            imgs = []
            for visual in visuals:
                if isinstance(visual, str):
                    # Video path - extract frames
                    try:
                        frames = self.encode_video(visual, self.max_frames_num)
                        imgs.extend(frames)
                    except Exception as e:
                        eval_logger.error(f"Error decoding video {visual}: {str(e)}")
                        continue
                elif isinstance(visual, Image.Image):
                    # PIL Image
                    img = self.encode_image(visual)
                    imgs.append(img)

            if not imgs:
                eval_logger.warning(f"No visual could be loaded for doc_id={doc_id}, returning empty response.")
                res.append("")
                pbar.update(1)
                if self.continual_mode is True:
                    doc_uuid = f"{task}___{split}___{doc_id}"
                    self.response_cache[doc_uuid] = ""
                    with open(self.response_persistent_file, "w") as f:
                        json.dump(self.response_cache, f)
                continue

            payload = {"messages": []}
            if API_TYPE == "openai":
                payload["model"] = self.model_version

            response_json = {"role": "user", "content": []}
            # When there is no image token in the context, append the image to the text
            if self.image_token not in contexts:
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][0]["content"].append({"type": "text", "text": contexts})
                for img in imgs:
                    payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
            else:
                contexts = contexts.split(self.image_token)
                for idx, img in enumerate(imgs):
                    payload["messages"].append(deepcopy(response_json))
                    payload["messages"][idx]["content"].append({"type": "text", "text": contexts[idx]})
                    payload["messages"][idx]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

                # If n image tokens are in the contexts
                # contexts will be splitted into n+1 chunks
                # Manually add it into the payload
                payload["messages"].append(deepcopy(response_json))
                payload["messages"][-1]["content"].append({"type": "text", "text": contexts[-1]})

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if gen_kwargs["max_new_tokens"] > 4096:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            payload["max_tokens"] = gen_kwargs["max_new_tokens"]
            payload["temperature"] = gen_kwargs["temperature"]

            for attempt in range(5):
                try:
                    response = url_requests.post(API_URL, headers=headers, json=payload, timeout=self.timeout)
                    response_data = response.json()

                    response_text = response_data["choices"][0]["message"]["content"].strip()
                    break  # If successful, break out of the loop

                except Exception as e:
                    try:
                        error_msg = response.json()
                    except:
                        error_msg = ""

                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}.\nReponse: {error_msg}")
                    if attempt < 4:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty string
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}.\nResponse: {error_msg}")
                        response_text = ""
            res.append(response_text)
            pbar.update(1)

            if self.continual_mode is True:  # Cache the response
                doc_uuid = f"{task}___{split}___{doc_id}"
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for GPT4O")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "GPT4O does not support loglikelihood"
