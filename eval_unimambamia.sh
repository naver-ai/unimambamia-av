#!/bin/bash
# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# UniMambaMia-AV Evaluation Script (lmms-eval)
#
# Usage:
#   ./eval_unimambamia.sh gwkrsrch/UniMambaMia-AV-Qwen2-7B
#   ./eval_unimambamia.sh /path/to/checkpoint --tasks av_speakerbench_audiovisual,avqa_hard
#   ./eval_unimambamia.sh /path/to/checkpoint --no_audio          # muted-video evaluation (audio ablation)
#   ./eval_unimambamia.sh /path/to/checkpoint --filtered_scores   # + scores on the single-frame filtered subsets (paper "filtered")
#   ./eval_unimambamia.sh /path/to/checkpoint --hf_home /path/to/cache --openai_key sk-xxx --num_gpus 8
#
# Notes:
#   * The released model was trained with 32 frames @ 1.0 fps, no time instruction, qwen_2 template.
#   * music_avqa / music_avqa_hard use a GPT judge (gpt-4o-mini) -> OPENAI_API_KEY is required.
#   * Benchmark videos are downloaded to $HF_HOME/<cache_dir> by lmms-eval (see each task yaml).

set -e

# ============================================
# Arguments
# ============================================
MODEL_PATH="${1:-gwkrsrch/UniMambaMia-AV-Qwen2-7B}"
shift 1 2>/dev/null || true

HF_HOME_ARG=""
OPENAI_KEY_ARG=""
USE_AUDIO=True
NUM_GPUS=8
MAXFRAME=32
MAXFPS=1.0
TEMPLATE="qwen_2"
ADDTIMEPROMPT=False
LOG_PATH="./eval_logs/"
FILTERED_SCORES=False
# Audio-visual benchmarks (paper Table 3):
#   av_speakerbench_audiovisual : AV-SpeakerBench
#   avqa_2025 / avqa_hard       : AVQA (re-packaged) / AVQA-Hard (single-frame filtered)
#   music_avqa / music_avqa_hard: Music-AVQA / Music-AVQA-Hard (GPT judge)
#   worldsense                  : WorldSense
# Vision-centric benchmarks (pinned lmms-eval):
#   videomme, longvideobench_val_v, activitynetqa, nextqa_mc_test, tempcompass_multi_choice, video_mmmu, vnbench_circular
BENCHLIST="av_speakerbench_audiovisual,avqa_hard,music_avqa_hard,worldsense"

while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)       BENCHLIST="$2"; shift 2 ;;
        --hf_home)     HF_HOME_ARG="$2"; shift 2 ;;
        --openai_key)  OPENAI_KEY_ARG="$2"; shift 2 ;;
        --num_gpus)    NUM_GPUS="$2"; shift 2 ;;
        --max_frames)  MAXFRAME="$2"; shift 2 ;;
        --fps)         MAXFPS="$2"; shift 2 ;;
        --log_path)    LOG_PATH="$2"; shift 2 ;;
        --no_audio)    USE_AUDIO=False; shift 1 ;;
        --filtered_scores) FILTERED_SCORES=True; shift 1 ;;   # also report scores on the single-frame filtered subsets
        *)
            echo "Unknown option: $1"
            echo "Usage: ./eval_unimambamia.sh [model_path_or_hf_id] [--tasks t1,t2] [--no_audio] [--filtered_scores] [--hf_home path] [--openai_key key] [--num_gpus N]"
            exit 1
            ;;
    esac
done

if [ -n "$HF_HOME_ARG" ]; then
    export HF_HOME="$HF_HOME_ARG"
    echo ">>> HF_HOME set to: $HF_HOME"
fi
if [ -n "$OPENAI_KEY_ARG" ]; then
    export OPENAI_API_KEY="$OPENAI_KEY_ARG"
    echo ">>> OPENAI_API_KEY set (hidden)"
fi

export DECORD_EOF_RETRY_MAX=20480
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p $LOG_PATH

echo "============================================"
echo "UniMambaMia-AV Evaluation Configuration"
echo "============================================"
echo "Model Path:      ${MODEL_PATH}"
echo "Use Audio:       ${USE_AUDIO}"
echo "Conv Template:   ${TEMPLATE}"
echo "Max Frames/FPS:  ${MAXFRAME} @ ${MAXFPS}"
echo "Benchmarks:      ${BENCHLIST}"
echo "HF_HOME:         ${HF_HOME:-[NOT SET]}"
if [ -n "$OPENAI_API_KEY" ]; then echo "OPENAI_API_KEY:  [SET]"; else echo "OPENAI_API_KEY:  [NOT SET] (needed for music_avqa*)"; fi
echo "============================================"

# ============================================
# Run Evaluation
# ============================================
# Note: install.sh installs unimambamia_vid.py and the audio-visual tasks into lmms-eval.
accelerate launch --num_machines 1 --num_processes $NUM_GPUS --main_process_port 12346 \
    -m lmms_eval \
    --model unimambamia_vid \
    --model_args pretrained=$MODEL_PATH,use_audio=$USE_AUDIO,max_frames_num=$MAXFRAME,video_fps=$MAXFPS,conv_template=$TEMPLATE,attn_implementation=flash_attention_2,add_time_instruction=$ADDTIMEPROMPT \
    --tasks $BENCHLIST \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix unimambamia_av_eval \
    --output_path $LOG_PATH

echo "============================================"
echo "Evaluation completed!"
echo "Results saved to: ${LOG_PATH}"
echo "============================================"

# ============================================
# Filtered scores (benchmark audit)
# ============================================
# Recompute every metric after removing the items GPT-4o answers from a single muted frame
# (benchmark_audit/single_frame_filter.json). Works from the sample logs written above; no re-inference.
if [ "$FILTERED_SCORES" = "True" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    MODEL_TAG=$(basename "$MODEL_PATH")
    python "$SCRIPT_DIR/benchmark_audit/recompute_filtered_scores.py" \
        --filter_json "$SCRIPT_DIR/benchmark_audit/single_frame_filter.json" \
        --log_dir "$LOG_PATH" --pattern "$MODEL_TAG" --show_original
fi
