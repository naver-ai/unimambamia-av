# Benchmark audit

This directory holds the material behind Section 3.1 of the paper: the GPT-4o single-frame probe,
the refusal-aware filter, the filter lists we release, the score recomputation script, and the
scripts that built AVQA-Hard and Music-AVQA-Hard.

## Protocol

An item is *single-frame answerable* if GPT-4o answers it correctly when shown only the
temporally central frame of the video, muted, with no other frames.

| Setting | Value |
|---------|-------|
| Model | `gpt-4o` (`gpt-4o-2024-08-06` at the time of the runs) |
| Visual input | one central frame, longer side resized to at most 384 px, JPEG |
| Prompt | the benchmark's own prompt as given to the Video-LLMs (via lmms-eval) |
| Runs | temperature 0 and temperature 1.0 |
| Criterion | correct in **both** runs, and the response is not a refusal |

The refusal check matters. The multiple-choice parsers of lmms-eval extract option letters from
free text, so a refusal such as "I Can't determine ..." is read as "C" and counted correct whenever
the answer happens to be C. Without the check AV-SpeakerBench appeared 25% single-frame answerable;
with it, 1.4%.

The filter is deliberately conservative. It removes only items that are demonstrably solvable from
a muted frame, which makes the filtered scores a lower bound on how much audio matters.

## Files

| File | Purpose |
|------|---------|
| `single_frame_filter.json` | The released filter lists, `{task: [doc_id, ...]}`. `doc_id` is the 0-based index of the item in the benchmark's evaluation split, as written by lmms-eval. The file covers the ten benchmarks of the paper as 15 lmms-eval tasks, plus VNBench, which was probed but not reported. |
| `../UniMambaMia/gpt4o_single_frame.py` | lmms-eval model `gpt4o`: sends the central frame to the OpenAI API, caches responses per rank (`continual_mode`). Installed by `install.sh`. |
| `generate_single_frame_filter.py` | Builds the filter list from the two probe runs (refusal-aware). |
| `recompute_filtered_scores.py` | Recomputes every benchmark metric on the filtered subset from existing `--log_samples` logs; no re-inference. |
| `hard_splits/probe_avqa_single_frame.py` | Central-frame GPT-4o probe for AVQA (multiple choice). |
| `hard_splits/probe_music_avqa_single_frame.py` | Central-frame GPT-4o probe for Music-AVQA (open-ended) with a gpt-4o-mini judge. |
| `hard_splits/build_hard_splits.py` | Builds AVQA-Hard / Music-AVQA-Hard from the probe outputs. |

Filtered items per task (`single_frame_filter.json`):

| Task | Items | Filtered | Ratio |
|------|------:|---------:|------:|
| tempcompass_multi_choice | 1,580 | 1,256 | 79.5% |
| avqa_2025 | 9,167 | 6,973 | 76.1% |
| music_avqa | 9,185 | 4,152 | 45.2% |
| videomme | 2,700 | 969 | 35.9% |
| activitynetqa | 8,000 | 2,501 | 31.3% |
| nextqa_mc_test | 8,564 | 2,654 | 31.0% |
| video_mmmu (perception / comprehension / adaptation) | 900 | 276 | 30.7% |
| longvideobench_val_v | 1,337 | 321 | 24.0% |
| avqa_hard | 1,696 | 276 | 16.3% |
| vnbench_circular | 5,400 | 243 | 4.5% |
| worldsense | 3,172 | 128 | 4.0% |
| music_avqa_hard | 2,514 | 77 | 3.1% |
| av_speakerbench_audiovisual | 3,212 | 44 | 1.4% |

## Reproducing the audit

```bash
export OPENAI_API_KEY=...
TASKS=avqa_hard,avqa_2025,av_speakerbench_audiovisual,activitynetqa,music_avqa_hard,music_avqa,worldsense,video_mmmu,videomme,longvideobench_val_v,nextqa_mc_test,tempcompass_multi_choice

# run 1: temperature 0
accelerate launch --num_processes 8 -m lmms_eval --model gpt4o \
    --model_args model_version=gpt-4o-2024-08-06,max_frames_num=1,continual_mode=True,response_persistent_folder=./logs/gpt4o_1frame_t0 \
    --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix gpt4o_1frame --output_path ./eval_logs/gpt4o_single_frame

# run 2: temperature 1.0
accelerate launch --num_processes 8 -m lmms_eval --model gpt4o \
    --model_args model_version=gpt-4o-2024-08-06,max_frames_num=1,continual_mode=True,response_persistent_folder=./logs/gpt4o_1frame_t1 \
    --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix gpt4o_1frame_temp1.0 --output_path ./eval_logs/gpt4o_single_frame \
    --gen_kwargs temperature=1.0

# filter list from the two runs (timestamps are the <YYYYMMDD_HHMMSS> prefixes of the sample files)
python benchmark_audit/generate_single_frame_filter.py --log_dir ./eval_logs/gpt4o_single_frame/<model_dir> \
    --temp0_timestamps <ts_run1> --temp1_timestamps <ts_run2> --output my_single_frame_filter.json
```

Music-AVQA and ActivityNet-QA use GPT judges (`gpt-4o-mini` and `gpt-3.5-turbo-0125`), so those
probe runs also need `OPENAI_API_KEY`. GPT-4o is stochastic even at temperature 0, so a regenerated list will differ slightly from
`single_frame_filter.json`. Use the released file to reproduce the numbers in the paper.

## Filtered scores for any model

Evaluate normally with `--log_samples`, then recompute on the filtered subset:

```bash
bash eval_unimambamia.sh gwkrsrch/UniMambaMia-AV-Qwen2-7B --tasks $TASKS --log_path ./eval_logs
python benchmark_audit/recompute_filtered_scores.py \
    --filter_json benchmark_audit/single_frame_filter.json \
    --log_dir ./eval_logs --pattern UniMambaMia --show_original
```

The script prints original and filtered scores side by side and can write CSV/JSON
(`--output_csv`, `--output_json`). Seed groups (`mean +- std`) are supported through
`--model_config` (see the docstring).

## Hard splits (AVQA-Hard, Music-AVQA-Hard)

The Hard splits were built in September 2025 with the same central-frame probe, at temperature 0
and in a single run, and released on the Hub:

* **AVQA-Hard** (`gwkrsrch/avqa_hard`, 1,696 items): items of `gwkrsrch/avqa_2025` (the AVQA
  validation items whose clips could still be retrieved) that GPT-4o answered incorrectly from
  the central frame.
* **Music-AVQA-Hard** (`gwkrsrch/music_avqa_hard`, 2,514 items): Music-AVQA test items whose
  question type involves audio (the `type` field does not contain `"Visual"`) and whose central-frame
  answer was judged wrong by gpt-4o-mini (`score 0`, `pred no`).

```bash
export OPENAI_API_KEY=...
python benchmark_audit/hard_splits/probe_avqa_single_frame.py --dataset gwkrsrch/avqa_2025 \
    --video-root $HF_HOME/avqa_cropped_videos_loadable --output avqa_probe.jsonl
python benchmark_audit/hard_splits/build_hard_splits.py avqa --probe avqa_probe.jsonl --output_dir ./avqa_hard

python benchmark_audit/hard_splits/probe_music_avqa_single_frame.py --video-root $HF_HOME/music_avqa --output music_probe.jsonl
python benchmark_audit/hard_splits/build_hard_splits.py music --probe music_probe.jsonl --output_dir ./music_avqa_hard
```

The released splits contain item identifiers (AVQA `id`; Music-AVQA `source_index`, the position of
the item in the official `avqa-test.json`) and the probe outputs; the scripts join the official
questions and answers through `UniMambaMia/official_annotations.py`, and `build_hard_splits.py`
writes the same identifier-only format.

The audit above re-probes both Hard splits with two temperature runs. The items that still turn out
to be single-frame answerable, 16.3% of AVQA-Hard and 3.1% of Music-AVQA-Hard, are what
`single_frame_filter.json` removes when those splits are evaluated in filtered mode.
