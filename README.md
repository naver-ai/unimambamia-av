<div align="center">

# Do Modern Video-LLMs Need to Listen? <br> A Benchmark Audit and Scalable Remedy

[![Paper](https://img.shields.io/badge/arXiv-2509.17901-b31b1b.svg)](https://arxiv.org/abs/2509.17901)
[![Conference](https://img.shields.io/badge/Interspeech-2026-4b44ce.svg)](https://www.interspeech2026.org/)
[![Model](https://img.shields.io/badge/HuggingFace-UniMambaMia--AV--Qwen2--7B-yellow.svg)](https://huggingface.co/gwkrsrch/UniMambaMia-AV-Qwen2-7B)
[![Data](https://img.shields.io/badge/HuggingFace-AVQA--Hard%20%7C%20Music--AVQA--Hard-yellow.svg)](https://huggingface.co/datasets/gwkrsrch/avqa_hard)

<img width="800" alt="main results" src="main_table.png">

</div>

## Introduction

Speech and audio encoders are routinely excluded from video understanding pipelines. We ask whether
that is a property of the models or of the benchmarks that certify them.

We first audit ten video benchmarks with a single-frame probe: GPT-4o is shown one muted frame from
the temporal centre of each clip. It answers about 76% of AVQA and 80% of TempCompass this way, so
on those suites a model can ignore audio without penalty. We then attach a speech/audio encoder to
a LLaVA-style Video-LLM and compare three input policies and five audio compressors under 25x token
reduction. Audio helps consistently on the suites that survive the audit, and the compressor we
adopt, UniMambaMia, is both the most stable of the five and the only design compatible with
streaming inference.

This repository contains the model, the training and evaluation code, the audit tooling with the
filter lists we release, and the scripts that built the AVQA-Hard and Music-AVQA-Hard splits.

> [Do Modern Video-LLMs Need to Listen? A Benchmark Audit and Scalable Remedy](https://arxiv.org/abs/2509.17901) <br>
> [Geewook Kim](https://geewook.kim) and [Minjoon Seo](https://scholar.google.com/citations?user=zYze5fIAAAAJ) <br>
> Interspeech 2026

## Updates

- 2026-09: Code, model and audit tooling released.
- 2025-11-24: AVQA-Hard and Music-AVQA-Hard released on the Hugging Face Hub in lmms-eval format.
- 2025-09-22: Preprint released on arXiv.

## Model

| Model | LLM | Vision encoder | Audio encoder | Audio compressor | Link |
|-------|-----|----------------|---------------|------------------|------|
| UniMambaMia-AV-Qwen2-7B | Qwen2-7B | SigLIP2-so400m, 384 px | Qwen2-Audio encoder (frozen) | UniMambaMia, 2 layers, 25x | [gwkrsrch/UniMambaMia-AV-Qwen2-7B](https://huggingface.co/gwkrsrch/UniMambaMia-AV-Qwen2-7B) |

The model reads 32 frames at 1 fps, which gives 144 visual tokens per frame after 2x2 pooling, and
one audio token per second of video. Audio tokens are placed directly after the visual tokens of
the frame they belong to.

## Datasets and filter lists

| Resource | Description | Link |
| :--- | :--- | :--- |
| AVQA-Hard | AVQA items that GPT-4o cannot answer from a single muted frame (1,696 items). | [gwkrsrch/avqa_hard](https://huggingface.co/datasets/gwkrsrch/avqa_hard) |
| Music-AVQA-Hard | Audio-related Music-AVQA items that survive the same probe (2,514 items). | [gwkrsrch/music_avqa_hard](https://huggingface.co/datasets/gwkrsrch/music_avqa_hard) |
| AVQA | AVQA validation items whose clips are still retrievable (9,167 items). | [gwkrsrch/avqa_2025](https://huggingface.co/datasets/gwkrsrch/avqa_2025) |
| Music-AVQA | Music-AVQA test split (9,185 items). | [gwkrsrch/music_avqa](https://huggingface.co/datasets/gwkrsrch/music_avqa) |
| Single-frame filter lists | The `doc_id`s judged single-frame answerable on all ten benchmarks (19,870 items). | [`benchmark_audit/single_frame_filter.json`](benchmark_audit/single_frame_filter.json) |

How these were built is documented in [`benchmark_audit/README.md`](benchmark_audit/README.md).

The four Hub datasets hold item identifiers and our probe outputs only. The questions, options and
answers belong to the original benchmarks and are not redistributed: at evaluation time
[`UniMambaMia/official_annotations.py`](UniMambaMia/official_annotations.py) downloads them from
the official AVQA and Music-AVQA repositories at a fixed commit, checks their SHA-256 and joins
them onto the released rows. No video is redistributed either; see [Evaluation data](#evaluation-data).

## Installation

```bash
git clone --recursive https://github.com/naver-ai/unimambamia-av
cd unimambamia-av
bash install.sh
```

`install.sh` checks out the pinned LLaVA and lmms-eval submodules, copies the sources in
`UniMambaMia/` over them, registers the two lmms-eval models (`unimambamia_vid` for our model,
`gpt4o` for the single-frame probe) together with the benchmark tasks that the pinned lmms-eval
does not carry, and installs the pinned dependencies.

The tested environment is Python 3.10, PyTorch 2.4 and CUDA 12.1; the flash-attn, mamba-ssm and
causal-conv1d wheels installed by the script target that combination. `ffmpeg` is required, since
both `decord` and the audit scripts decode audio through it.

## Evaluation

```bash
# audio-visual benchmarks (default: AV-SpeakerBench, AVQA-Hard, Music-AVQA-Hard, WorldSense)
bash eval_unimambamia.sh gwkrsrch/UniMambaMia-AV-Qwen2-7B --hf_home /path/to/hf_cache --openai_key sk-...

# any lmms-eval task list, for example the vision-centric suites
bash eval_unimambamia.sh gwkrsrch/UniMambaMia-AV-Qwen2-7B --tasks videomme,longvideobench_val_v,tempcompass_multi_choice,video_mmmu

# the same checkpoint on muted video (the audio ablation of Table 1)
bash eval_unimambamia.sh gwkrsrch/UniMambaMia-AV-Qwen2-7B --no_audio

# additionally report scores on the single-frame filtered subsets
bash eval_unimambamia.sh gwkrsrch/UniMambaMia-AV-Qwen2-7B --tasks av_speakerbench_audiovisual,avqa_2025 --filtered_scores
```

The `unimambamia_vid` model accepts the following arguments; the defaults match the released model.

| Argument | Description | Default |
|----------|-------------|---------|
| `use_audio` | Feed the soundtrack, or evaluate the muted video. | `True` |
| `max_frames_num`, `video_fps` | Visual frame budget. | `32`, `1.0` |
| `conv_template` | Conversation template. | `qwen_2` |
| `add_time_instruction` | Prepend "The video lasts for ..." to the prompt. | `False` |

Music-AVQA and ActivityNet-QA are scored by GPT judges, so those tasks need `OPENAI_API_KEY`.
For most benchmarks lmms-eval downloads the videos into `$HF_HOME/<cache_dir>`; AVQA and
Music-AVQA are the exception (see below).

Filtered scores are recomputed from the sample logs by
[`benchmark_audit/recompute_filtered_scores.py`](benchmark_audit/recompute_filtered_scores.py), so
any model evaluated with `--log_samples` can be scored on the filtered subsets without running
inference again.

### Evaluation data

The videos of AVQA and Music-AVQA are not hosted with this repository and have to be obtained
from the original projects under their terms:

| Task | Videos | Expected location |
| :--- | :--- | :--- |
| `avqa_2025`, `avqa_hard` | [AVQA](https://mn.cs.tsinghua.edu.cn/avqa/) clips (drawn from VGGSound), one file per `video_name` | `$HF_HOME/avqa_cropped_videos_loadable/<video_name>.mp4` |
| `music_avqa`, `music_avqa_hard` | [Music-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/) videos | `$HF_HOME/music_avqa/<video_id>.mp4` |

The annotation files are fetched automatically on first use and cached under
`$HF_HOME/unimambamia_av/official`. On machines without internet access, download
[`val_qa.json`](https://github.com/AlyssaYoung/AVQA/blob/5f5b87157e5265e85d88ae22a9bee03046f45fb6/data/annotation/val_qa.json) and
[`avqa-test.json`](https://github.com/GeWu-Lab/MUSIC-AVQA/blob/f72f0e8a9bb897826aa0dda934fdece339d4e62f/data/json/avqa-test.json)
beforehand and set `UNIMAMBAMIA_AVQA_ANNOTATIONS` and `UNIMAMBAMIA_MUSIC_AVQA_ANNOTATIONS` to their paths.

## Training

Training starts from an image-level instruction-tuned LLaVA-style VLM (SigLIP2 with Qwen2-7B,
trained following [ELVA](https://github.com/naver-ai/elva)) and proceeds in two stages. In the
first stage the frozen audio encoder is attached and only the vision projector and the audio
compressor are trained, on audio-visual captions and speech transcripts. In the second stage the
LLM is unfrozen, with both encoders kept frozen, and trained on the audio-visual instruction mix:
LLaVA-Video-178K subsets, FineVideo, the Music-AVQA v2 training set, AVSD and the AVQA training
set.

```bash
./train_unimambamia.sh \
  --vlm_path /path/to/image_instruction_tuned_vlm \
  --align_jsonl  "/data/llava_video/0_30_s_youtube_cap.jsonl,/data/finevideo/stt_under_500_words.jsonl" \
  --align_folder "/data/llava_video,/data/finevideo" \
  --video_jsonl  "/data/avqa/mcqa.jsonl,/data/music_avqa/train_v2.jsonl,/data/llava_video/2_3_m_youtube_cap.jsonl" \
  --video_folder "/data/avqa,/data/music_avqa,/data/llava_video" \
  --num_gpus 8 --num_nodes 1
```

Training data are LLaVA-style JSONL files whose `video` field names a file inside the matching
folder; `convert_json_to_jsonl.py` converts LLaVA JSON. The soundtrack is decoded from the video
file itself, so the video folders have to hold the media files rather than pre-extracted frames.

| Argument | Description | Released model |
|----------|-------------|----------------|
| `--audio_projector_type` | `audio_projector_mambamia2uni` (UniMambaMia), `audio_projector_mamba2uni` (UniMamba), `audio_projector_mamba2bi` (BiMamba), `audio_projector_resampler` (Resampler), `audio_{R}patchwise_mlp2x_gelu` (Avg Pool) | `audio_projector_mambamia2uni` |
| `--compression_ratio` | Compression ratio R, that is one query every R encoder tokens. | `25` |
| `--audio_layers`, `--audio_init_scale` | Layers and gated-attention initialization scale of the compressor. | `2`, `0.001` |
| `--merge_type` | `..._avepool2_zigzag` for time-aligned interleaving, `..._avepool2` for non-interleaving. | `..._avepool2_zigzag` |
| `--max_frame`, `--fps` | Visual frame budget. | `32`, `1.0` |

The released model was trained with an effective batch of 64 samples and a learning rate of 5e-5 in
stage 1, and 240 samples at 2.5e-5 with `model_max_length` 64000 in stage 2. `--stage 1` and
`--stage 2` run a single stage; `--pretrain_mm_mlp_adapter` passes the stage-1 projector file,
which holds both the vision projector and the audio compressor.

## Method

```
video --> SigLIP2 (frozen) --> MLP --> 2x2 avg-pool --> 144 tokens per frame ----------+
                                                                                       +--> per-frame interleaving --> Qwen2-7B
audio --> Qwen2-Audio encoder (frozen, 25 Hz) --> UniMambaMia (25x) --> 1 token per s --+
```

The audio compressor is implemented in [`UniMambaMia/audio_compressor.py`](UniMambaMia/audio_compressor.py).
Encoder features are split into blocks of R tokens and one shared learnable query is appended to
each block. The whole block-and-query sequence of a clip is processed by a causal Mamba2 stack of
two layers with the gated pooling aggregation of MambaMia, in which each query is updated with a
learned, softmax-weighted average of its own block, gated against the query itself. Only the query
outputs are kept, which yields the R-fold reduction, and they are projected to the LLM width.
Because the stack is causal, the module can run incrementally as audio arrives.

The four other compressors compared in the paper use the same interface: `resampler` is a
bidirectional transformer with the same periodic queries, `mamba2uni` and `mamba2bi` drop the gated
aggregation, and the Avg Pool baseline replaces the module with an MLP followed by average pooling
every R tokens.

## Benchmark audit

`benchmark_audit/` holds the single-frame probe, the refusal-aware filter generator, the released
filter lists, the score recomputation script and the scripts that built the Hard splits. See
[`benchmark_audit/README.md`](benchmark_audit/README.md).

## Repository layout

```
UniMambaMia/            sources copied over LLaVA and lmms-eval by install.sh
  audio_compressor.py   periodic-query audio compressor (UniMambaMia and the baselines)
  modeling_mambamia2.py MambaMia2 backbone: causal, causal with gated aggregation, bidirectional
  llava_arch.py         audio tower, audio encoding and audio-visual token construction
  train.py              training loop and the audio-aware data pipeline
  video_utils.py        video and audio sampling (30 s waveform chunks -> log-Mel windows)
  unimambamia_vid.py    lmms-eval model wrapper
  gpt4o_single_frame.py lmms-eval GPT-4o single-frame probe
  lmms_eval_tasks/      benchmark tasks missing from the pinned lmms-eval
benchmark_audit/        filter lists, filter generator, score recomputation, Hard-split scripts
install.sh              submodules, source overlay and pinned dependencies
train_unimambamia.sh    two-stage training
eval_unimambamia.sh     lmms-eval evaluation, optionally with filtered scores
upload_to_hf.py         push a trained checkpoint to the Hugging Face Hub
```

Not included: the checkpoints of the compressor and input-policy baselines, which are reproduced by
retraining with the arguments above, and the image-level VLM that stage 1 starts from.

## Citation

```bibtex
@inproceedings{kim2026doesaudiomatter,
  title     = {Do Modern Video-LLMs Need to Listen? A Benchmark Audit and Scalable Remedy},
  author    = {Geewook Kim and Minjoon Seo},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026},
  url       = {https://arxiv.org/abs/2509.17901}
}
```

## License

The code is released under the MIT License, reproduced in [`LICENSE`](LICENSE). It builds on
MambaMia, LLaVA, LLaVA-NeXT, Transformers, lmms-eval, MMMU and FastChat, whose license
terms and the files they apply to are listed in [`NOTICE`](NOTICE). The released weights combine Qwen2-7B, SigLIP2 and the Qwen2-Audio encoder, all
under Apache 2.0; the training and evaluation datasets keep their own licenses. The four Hub
datasets listed above contain only item identifiers and values we produced (probe outputs, clip
durations), released under MIT; the AVQA and Music-AVQA annotations and videos they refer to are
not included and remain under the terms of their original releases.

```
UniMambaMia-AV
Copyright (c) 2026-present NAVER Cloud Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgements

This project builds upon:
- [MambaMia](https://github.com/naver-ai/mambamia), whose compressor the audio module is adapted from, and [ELVA](https://github.com/naver-ai/elva), our sister projects
- [LLaVA](https://github.com/haotian-liu/LLaVA) and [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Transformers](https://github.com/huggingface/transformers) and [Mamba](https://github.com/state-spaces/mamba)
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), the evaluation framework
- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio) and [SigLIP2](https://huggingface.co/google/siglip2-so400m-patch16-384) for the encoders
- [AV-SpeakerBench](https://huggingface.co/datasets/plnguyen2908/AV-SpeakerBench), [WorldSense](https://huggingface.co/datasets/lmms-lab/worldsense), [AVQA](https://mn.cs.tsinghua.edu.cn/avqa/), [Music-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/) and the other benchmarks evaluated in the paper
