# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Upload a trained UniMambaMia-AV checkpoint to the Hugging Face Hub.
#
# Usage:
#   hf auth login   # or export HF_TOKEN=...
#   python upload_to_hf.py /path/to/checkpoints/video_llava_<RUN_NAME> gwkrsrch/UniMambaMia-AV-Qwen2-7B
#   python upload_to_hf.py /path/to/ckpt gwkrsrch/UniMambaMia-AV-Qwen2-7B --private --dtype float16
#
# What it does:
#   1. (optional) re-saves the safetensors shards in float16 / bfloat16 to halve the upload size
#   2. strips training-only files (trainer_state.json, training_args.bin, optimizer/*)
#   3. writes a model card (README.md) and uploads everything with huggingface_hub

import argparse
import json
import os
import shutil
import tempfile

from huggingface_hub import HfApi

MODEL_CARD = """---
license: apache-2.0
library_name: transformers
pipeline_tag: video-text-to-text
tags:
  - video
  - audio
  - audio-visual
  - llava
  - mamba
  - unimambamia
base_model:
  - Qwen/Qwen2-7B
  - google/siglip2-so400m-patch16-384
  - Qwen/Qwen2-Audio-7B-Instruct
---

# {repo_name}

Audio-visual Video-LLM from *Do Modern Video-LLMs Need to Listen? A Benchmark Audit and Scalable
Remedy* (Interspeech 2026, [arXiv:2509.17901](https://arxiv.org/abs/2509.17901)).

| Component | |
| :--- | :--- |
| LLM | Qwen2-7B |
| Vision encoder | SigLIP2-so400m, 384 px, 144 tokens per frame after 2x2 pooling (frozen) |
| Audio encoder | Qwen2-Audio encoder (frozen) |
| Audio compressor | UniMambaMia (causal Mamba2 with gated pooling, 2 layers), 25x compression: one audio token per second |
| Input | 32 frames at 1 fps; the audio tokens of each second follow the visual tokens of that frame |

The weights are stored in float16, the precision used for the evaluations in the paper.

Code, training and evaluation scripts: https://github.com/naver-ai/unimambamia-av

## Usage

```bash
git clone --recursive https://github.com/naver-ai/unimambamia-av && cd unimambamia-av
bash install.sh
bash eval_unimambamia.sh {repo_id} --tasks av_speakerbench_audiovisual,avqa_hard
```

## License

Apache 2.0, following the licenses of Qwen2-7B, SigLIP2 and the Qwen2-Audio encoder that the
model is built from.

## Citation

```bibtex
@inproceedings{{kim2026doesaudiomatter,
  title     = {{Do Modern Video-LLMs Need to Listen? A Benchmark Audit and Scalable Remedy}},
  author    = {{Geewook Kim and Minjoon Seo}},
  booktitle = {{Proc. Interspeech 2026}},
  year      = {{2026}},
  url       = {{https://arxiv.org/abs/2509.17901}}
}}
```
"""

TRAINING_ONLY = {"trainer_state.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"}


def convert_dtype(src_dir, dst_dir, dtype):
    import torch
    from safetensors.torch import load_file, save_file

    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    shards = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = ["model.safetensors"]

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    total = 0
    for shard in shards:
        sd = load_file(os.path.join(src_dir, shard))
        sd = {k: (v.to(torch_dtype) if v.is_floating_point() else v) for k, v in sd.items()}
        total += sum(v.numel() * v.element_size() for v in sd.values())
        save_file(sd, os.path.join(dst_dir, shard), metadata={"format": "pt"})
        print(f"  converted {shard} -> {dtype}")
    if os.path.exists(index_path):
        index["metadata"] = {"total_size": total}
        with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    with open(os.path.join(src_dir, "config.json")) as f:
        config = json.load(f)
    config["torch_dtype"] = dtype
    with open(os.path.join(dst_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", help="local checkpoint directory (output of train_unimambamia.sh stage 2)")
    parser.add_argument("repo_id", help="e.g. gwkrsrch/UniMambaMia-AV-Qwen2-7B")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--dtype", choices=["keep", "bfloat16", "float16"], default="keep",
                        help="re-save weights in this dtype before uploading (fp32 checkpoints are ~33GB; 16-bit halves it)")
    parser.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN env / cached login)")
    parser.add_argument("--work_dir", default=None,
                        help="where to write the converted copy (default: next to the checkpoint)")
    parser.add_argument("--dry_run", action="store_true", help="prepare the folder but do not upload")
    args = parser.parse_args()

    api = HfApi(token=args.token)
    who = api.whoami()
    print(f"Logged in as: {who.get('name')}")

    ignore = list(TRAINING_ONLY) + ["checkpoint-*/**", "global_step*/**", "*.pt", "*.pth"]
    card = MODEL_CARD.format(repo_name=args.repo_id.split("/")[-1], repo_id=args.repo_id)

    if args.dtype == "keep":
        # Upload the checkpoint folder as-is (training-only files ignored) + a model card.
        print("Files to upload:")
        for name in sorted(os.listdir(args.ckpt_dir)):
            if name in TRAINING_ONLY or name.startswith("checkpoint-") or name.startswith("global_step"):
                continue
            print("  ", name)
        if args.dry_run:
            print("--dry_run: skipping upload")
            return
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="model", folder_path=args.ckpt_dir, ignore_patterns=ignore)
        api.upload_file(repo_id=args.repo_id, repo_type="model", path_or_fileobj=card.encode(), path_in_repo="README.md")
        print(f"Done: https://huggingface.co/{args.repo_id}")
        return

    # dtype conversion: materialize a converted copy (next to the checkpoint by default), then upload it.
    work_dir = args.work_dir or os.path.dirname(os.path.abspath(args.ckpt_dir))
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        print(f"Preparing upload folder at {tmp}")
        for name in os.listdir(args.ckpt_dir):
            src = os.path.join(args.ckpt_dir, name)
            if name in TRAINING_ONLY or name.startswith("checkpoint-") or name.startswith("global_step") or os.path.isdir(src):
                continue
            if name.endswith(".safetensors") or name in ("model.safetensors.index.json", "config.json"):
                continue  # written by convert_dtype
            shutil.copy2(src, os.path.join(tmp, name))
        convert_dtype(args.ckpt_dir, tmp, args.dtype)
        with open(os.path.join(tmp, "README.md"), "w") as f:
            f.write(card)

        print("Files to upload:")
        for name in sorted(os.listdir(tmp)):
            print("  ", name)
        if args.dry_run:
            print("--dry_run: skipping upload")
            return

        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_large_folder(repo_id=args.repo_id, repo_type="model", folder_path=tmp)
        print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
