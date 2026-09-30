# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Copied from https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/train/train_mem.py

from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
