# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Modified from https://github.com/naver-ai/mambamia/blob/main/MambaMia/llava_qwen.py,
# which is modified from
# https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/model/language_model/llava_qwen.py
# Below is the original copyright:
#
#    Copyright 2024 Hao Zhang
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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

from llava.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config, delay_load=True):
        super(LlavaQwenModel, self).__init__(config, delay_load=delay_load)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    """Qwen2 backbone with a vision tower, an audio tower and their projectors."""

    config_class = LlavaQwenConfig

    def __init__(self, config, delay_load=True):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config, delay_load=delay_load)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        audio_masks: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                images=images,
                audio_values=audio_values,
                audio_masks=audio_masks,
            )

        if labels is None:
            # Inference: the standard Qwen2 forward.
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Training: fused linear cross entropy (Liger) keeps the logits out of memory, which matters
        # at the sequence lengths reached once audio tokens are interleaved.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = None

        shift_labels = labels[..., 1:].contiguous().view(-1)
        hidden_states = hidden_states[..., :-1, :].contiguous()
        hidden_states = hidden_states.view(-1, self.config.hidden_size).to(self.lm_head.weight.dtype)
        loss = LigerFusedLinearCrossEntropyLoss()(self.lm_head.weight, hidden_states, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.FloatTensor] = None,
        audio_masks: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            inputs, position_ids, attention_mask, _, inputs_embeds, _ = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                audio_values=audio_values,
                audio_masks=audio_masks,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        audio_values = kwargs.pop("audio_values", None)
        audio_masks = kwargs.pop("audio_masks", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if audio_values is not None:
            inputs["audio_values"] = audio_values
        if audio_masks is not None:
            inputs["audio_masks"] = audio_masks
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
