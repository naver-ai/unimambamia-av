#!/bin/bash
# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# UniMambaMia-AV Training Script (Qwen2-7B backbone, SigLIP2 vision encoder, Qwen2-Audio encoder)
#
# The recipe follows the paper ("Do Modern Video-LLMs Need to Listen?", Sec. 4.1):
#   0) Start from an image-level instruction-tuned LLaVA-style VLM (SigLIP2 + Qwen2-7B, trained following ELVA).
#   1) Stage 1 - module-only alignment: attach the frozen audio encoder and train ONLY the vision MLP projector and
#      the UniMambaMia audio compressor on audio-visual captions / speech transcripts (LLaVA-Video + FineVideo).
#   2) Stage 2 - video instruction tuning: unfreeze the LLM (both encoders stay frozen) and train on the
#      audio-visual instruction mix (LLaVA-Video-178K subsets, FineVideo, Music-AVQA v2, AVSD/Charades-STA, AVQA).
#
# ============================================
# Usage:
# ============================================
#   Minimal example (stage 1 + stage 2):
#     ./train_unimambamia.sh \
#       --vlm_path /path/to/image_instruction_tuned_vlm \
#       --align_jsonl  "/data/LLaVA-Video-178K/0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_cap_processed.jsonl,/data/finevideo/stt_under_500_words.jsonl" \
#       --align_folder "/data/LLaVA-Video-178K/0_30_s_youtube_v0_1,/data/finevideo" \
#       --video_jsonl  "/data/avqa/mcqa.jsonl,/data/music_avqa/train_v2.jsonl,/data/LLaVA-Video-178K/2_3_m_youtube_v0_1/2_3_m_youtube_v0_1_cap_processed.jsonl" \
#       --video_folder "/data/avqa,/data/music_avqa,/data/LLaVA-Video-178K/2_3_m_youtube_v0_1"
#
#   Run only one stage:
#     ./train_unimambamia.sh --stage 1 ...      # alignment only  -> ./checkpoints/pretrain_video_llava_$RUN_NAME/mm_projector.bin
#     ./train_unimambamia.sh --stage 2 --pretrain_mm_mlp_adapter ./checkpoints/pretrain_video_llava_$RUN_NAME/mm_projector.bin ...
#
#   Compressor comparison (paper Table 2):
#     --audio_projector_type audio_projector_mambamia2uni   # UniMambaMia (default, ours)
#     --audio_projector_type audio_projector_mamba2uni      # UniMamba
#     --audio_projector_type audio_projector_mamba2bi       # BiMamba
#     --audio_projector_type audio_projector_resampler      # Resampler
#     --audio_projector_type audio_25patchwise_mlp2x_gelu   # Avg Pool (25x)
#     --compression_ratio 5|10|25                           # R in "{R}patchwise" (default 25 = 1 audio token / second)
#
#   Input policy ablations (paper Table 1):
#     --merge_type flat_video_frame_with_end_token_and_avepool2_zigzag   # time-aligned interleaving (default)
#     --merge_type flat_video_frame_with_end_token_and_avepool2          # non-interleaving ([V; A])
#     --no_audio                                                        # vision-only
#
# ============================================
# Required Options:
# ============================================
#   --vlm_path PATH        Image-level instruction-tuned LLaVA-style checkpoint (SigLIP2 + Qwen2-7B)
#   --align_jsonl PATH     Stage-1 JSONL file(s), comma-separated          (skip with --stage 2)
#   --align_folder PATH    Stage-1 video folder(s), comma-separated         (skip with --stage 2)
#   --video_jsonl PATH     Stage-2 JSONL file(s), comma-separated           (skip with --stage 1)
#   --video_folder PATH    Stage-2 video folder(s), comma-separated         (skip with --stage 1)
#   Note: the number of jsonl files and folders must match (one folder per jsonl).
#
# ============================================
# Optional Options:
# ============================================
#   --stage {1,2,all}            Which stage(s) to run (default: all)
#   --run_name NAME              Run name (default: unimambamia_av_siglip2_qwen2_7b)
#   --hf_home PATH               HuggingFace cache directory (export HF_HOME)
#   --clip_path PATH             Vision encoder (default: gwkrsrch2/siglip2-so400m-patch16-384)
#   --audio_path PATH            Audio encoder  (default: gwkrsrch2/qwen2-audio-encoder-from-qwen2-audio-7b-instruct)
#   --pretrain_mm_mlp_adapter P  Projector weights to initialize stage 1 (vision MLP) / stage 2 (vision MLP + audio compressor)
#   --num_gpus N                 GPUs per node (default: 8)
#   --num_nodes N                Number of nodes (default: 1); set MASTER_ADDR / NODE_RANK env for multi-node
#
#   Audio compressor:
#   --audio_projector_type T     Compressor (default: audio_projector_mambamia2uni)
#   --audio_layers N             Mamba layers in the compressor (default: 2)
#   --audio_init_scale F         Gated-attention init scale (default: 0.001)
#   --compression_ratio R        Audio compression ratio (default: 25)
#
#   Stage 1 (alignment) hyper-parameters (released model: 16 GPUs x 4 x 1 = 64 samples/step, lr 5e-5, 8 frames):
#   --align_batch N (4)  --align_accum N (1)  --align_lr F (5e-5)  --align_max_frame N (8)  --align_max_len N (24000)
#
#   Stage 2 (instruction tuning) hyper-parameters (released model: 48 GPUs x 1 x 5 = 240 samples/step, lr 2.5e-5, 32 frames):
#   --per_device N (1)  --accum N (5)  --lr F (2.5e-5)  --max_frame N (32)  --fps F (1.0)  --model_max_length N (64000)
#   --max_video_duration S (900)   Skip videos longer than S seconds (audio memory guard)
#   --seed N (202602)

set -e

export DECORD_EOF_RETRY_MAX=20480  # Prevent video decoding OOM
export TOKENIZERS_PARALLELISM=false
export DS_SKIP_CUDA_CHECK=1
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
# Optional: jemalloc reduces host-memory fragmentation from audio/video decoding workers
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
# export MALLOC_CONF="dirty_decay_ms:0,muzzy_decay_ms:0"

# ============================================
# Helper Functions
# ============================================
# Convert comma-separated paths to Python list format: "a.jsonl,b.jsonl" -> "['a.jsonl','b.jsonl']"
to_python_list() {
    local input="$1"
    echo "['"$(echo "$input" | sed "s/,/','/g")"']"
}

# ============================================
# Arguments Parsing
# ============================================
STAGE="all"
RUN_NAME="unimambamia_av_siglip2_qwen2_7b"
HF_HOME_ARG=""
VLM_PATH=""
CLIP_PATH="gwkrsrch2/siglip2-so400m-patch16-384"
AE_PATH="gwkrsrch2/qwen2-audio-encoder-from-qwen2-audio-7b-instruct"
PRETRAIN_MM_MLP_ADAPTER=""
ALIGN_JSONL=""
ALIGN_FOLDER=""
VIDEO_JSONL=""
VIDEO_FOLDER=""
NUM_GPUS=8
NUM_NODES=1

USE_AUDIO=True
AUDIO_PROJECTOR_TYPE="audio_projector_mambamia2uni"
AUDIO_PROJECTOR_NUM_LAYERS=2
AUDIO_PROJECTOR_INIT_SCALE=0.001
COMPRESSION_RATIO=25
MERGE_TYPE="flat_video_frame_with_end_token_and_avepool2_zigzag"

ALIGN_BATCH=4
ALIGN_ACCUM=1
ALIGN_LR=5e-5
ALIGN_MAX_FRAME=8
ALIGN_MAX_LEN=24000

PER_DEVICE=1
ACCUM=5
LR=2.5e-5
MAX_FRAME=32
FPS=1.0
MODEL_MAX_LENGTH=64000
MAX_VIDEO_DURATION=900
SEED=202602

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)                    STAGE="$2"; shift 2 ;;
        --run_name)                 RUN_NAME="$2"; shift 2 ;;
        --hf_home)                  HF_HOME_ARG="$2"; shift 2 ;;
        --vlm_path)                 VLM_PATH="$2"; shift 2 ;;
        --clip_path)                CLIP_PATH="$2"; shift 2 ;;
        --audio_path)               AE_PATH="$2"; shift 2 ;;
        --pretrain_mm_mlp_adapter)  PRETRAIN_MM_MLP_ADAPTER="$2"; shift 2 ;;
        --align_jsonl)              ALIGN_JSONL="$2"; shift 2 ;;
        --align_folder)             ALIGN_FOLDER="$2"; shift 2 ;;
        --video_jsonl)              VIDEO_JSONL="$2"; shift 2 ;;
        --video_folder)             VIDEO_FOLDER="$2"; shift 2 ;;
        --num_gpus)                 NUM_GPUS="$2"; shift 2 ;;
        --num_nodes)                NUM_NODES="$2"; shift 2 ;;
        --no_audio)                 USE_AUDIO=False; shift 1 ;;
        --audio_projector_type)     AUDIO_PROJECTOR_TYPE="$2"; shift 2 ;;
        --audio_layers)             AUDIO_PROJECTOR_NUM_LAYERS="$2"; shift 2 ;;
        --audio_init_scale)         AUDIO_PROJECTOR_INIT_SCALE="$2"; shift 2 ;;
        --compression_ratio)        COMPRESSION_RATIO="$2"; shift 2 ;;
        --merge_type)               MERGE_TYPE="$2"; shift 2 ;;
        --align_batch)              ALIGN_BATCH="$2"; shift 2 ;;
        --align_accum)              ALIGN_ACCUM="$2"; shift 2 ;;
        --align_lr)                 ALIGN_LR="$2"; shift 2 ;;
        --align_max_frame)          ALIGN_MAX_FRAME="$2"; shift 2 ;;
        --align_max_len)            ALIGN_MAX_LEN="$2"; shift 2 ;;
        --per_device)               PER_DEVICE="$2"; shift 2 ;;
        --accum)                    ACCUM="$2"; shift 2 ;;
        --lr)                       LR="$2"; shift 2 ;;
        --max_frame)                MAX_FRAME="$2"; shift 2 ;;
        --fps)                      FPS="$2"; shift 2 ;;
        --model_max_length)         MODEL_MAX_LENGTH="$2"; shift 2 ;;
        --max_video_duration)       MAX_VIDEO_DURATION="$2"; shift 2 ;;
        --seed)                     SEED="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./train_unimambamia.sh --vlm_path PATH [--stage 1|2|all] [OPTIONS]  (see script header)"
            exit 1
            ;;
    esac
done

# Validate
if [ -z "$VLM_PATH" ]; then
    echo "Error: --vlm_path is required (image-level instruction-tuned SigLIP2 + Qwen2-7B LLaVA checkpoint)."
    exit 1
fi
if [ "$STAGE" = "1" ] || [ "$STAGE" = "all" ]; then
    if [ -z "$ALIGN_JSONL" ] || [ -z "$ALIGN_FOLDER" ]; then
        echo "Error: --align_jsonl and --align_folder are required for stage 1."; exit 1
    fi
fi
if [ "$STAGE" = "2" ] || [ "$STAGE" = "all" ]; then
    if [ -z "$VIDEO_JSONL" ] || [ -z "$VIDEO_FOLDER" ]; then
        echo "Error: --video_jsonl and --video_folder are required for stage 2."; exit 1
    fi
fi
if [ -n "$HF_HOME_ARG" ]; then
    export HF_HOME="$HF_HOME_ARG"
    echo ">>> HF_HOME set to: $HF_HOME"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSDP_CONFIG="$SCRIPT_DIR/fsdp_qwen.yaml"
CONV_TEMPLATE="qwen_2"

# Vision side: SigLIP2 (384px, patch16 -> 24x24=576 tokens) -> MLP projector -> 2x2 average pooling (144 tokens/frame)
PROJECTOR_TYPE="mlp2x_gelu"
USE_LAYER=-2

# Audio side: R x compression via periodic queries (1 audio token / second when R=25 at the 25 Hz encoder rate)
AUDIO_PROJECTOR_MODE="per_from1to1frames_${COMPRESSION_RATIO}patchwise_${COMPRESSION_RATIO}tokperframe"
if [[ "$AUDIO_PROJECTOR_TYPE" == audio_*patchwise_mlp* ]]; then
    # Avg Pool baseline: the ratio is encoded in the projector type itself
    AUDIO_PROJECTOR_TYPE="audio_${COMPRESSION_RATIO}patchwise_mlp2x_gelu"
fi
if [ "$USE_AUDIO" = "False" ]; then
    AE_PATH=""
    MERGE_TYPE="${MERGE_TYPE%_zigzag}"
fi
AUDIO_ARGS=""
if [ "$USE_AUDIO" = "True" ]; then
    AUDIO_ARGS="--audio_tower $AE_PATH --use_audio True \
        --mm_audio_projector_type $AUDIO_PROJECTOR_TYPE \
        --audio_projector_mode $AUDIO_PROJECTOR_MODE \
        --audio_projector_num_layers $AUDIO_PROJECTOR_NUM_LAYERS \
        --audio_projector_init_scale $AUDIO_PROJECTOR_INIT_SCALE \
        --max_video_duration $MAX_VIDEO_DURATION \
        --freeze_two_encoder True"
fi

# Multi-node (torchrun-style env): MASTER_ADDR, MASTER_PORT, NODE_RANK
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROC=$((NUM_GPUS * NUM_NODES))
LAUNCH="accelerate launch --num_processes $NUM_PROC --num_machines $NUM_NODES --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --machine_rank $NODE_RANK --config_file $FSDP_CONFIG"

COMMON_ARGS="--vision_tower $CLIP_PATH \
    --version $CONV_TEMPLATE \
    --mm_projector_type $PROJECTOR_TYPE \
    --mm_vision_select_layer $USE_LAYER \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_grid_pinpoints \"[(1,1),]\" \
    --mm_patch_merge_type $MERGE_TYPE \
    --add_time_instruction False \
    --image_aspect_ratio square \
    --video_fps $FPS \
    --fp16 False --bf16 True --tf32 True \
    --torch_compile False \
    --num_train_epochs 1 \
    --evaluation_strategy no \
    --save_strategy steps --save_total_limit 1 \
    --adam_epsilon 1e-8 --max_grad_norm 2.0 --weight_decay 0.0 \
    --warmup_ratio 0.03 --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory False \
    --lazy_preprocess True \
    --report_to wandb"

echo "============================================"
echo "UniMambaMia-AV Training"
echo "============================================"
echo "Stage:              $STAGE"
echo "Run name:           $RUN_NAME"
echo "VLM init:           $VLM_PATH"
echo "Vision encoder:     $CLIP_PATH"
echo "Audio encoder:      ${AE_PATH:-[disabled]}"
echo "Audio compressor:   $AUDIO_PROJECTOR_TYPE ($AUDIO_PROJECTOR_MODE, layers=$AUDIO_PROJECTOR_NUM_LAYERS, init_scale=$AUDIO_PROJECTOR_INIT_SCALE)"
echo "Merge type:         $MERGE_TYPE"
echo "GPUs:               $NUM_NODES node(s) x $NUM_GPUS"
echo "============================================"

cd "$SCRIPT_DIR/LLaVA"
mkdir -p ./checkpoints

# ============================================
# Stage 1: module-only alignment (vision MLP + audio compressor)
# ============================================
STAGE1_OUT="./checkpoints/pretrain_video_llava_$RUN_NAME"
if [ "$STAGE" = "1" ] || [ "$STAGE" = "all" ]; then
    echo ">>> Stage 1: alignment  (effective batch = $ALIGN_BATCH x $NUM_PROC x $ALIGN_ACCUM)"
    ALIGN_INIT=""
    if [ -n "$PRETRAIN_MM_MLP_ADAPTER" ]; then
        # e.g. a video-pretrained vision MLP; missing audio weights are tolerated (ignore_mismatched_sizes)
        ALIGN_INIT="--pretrain_mm_mlp_adapter $PRETRAIN_MM_MLP_ADAPTER --ignore_mismatched_sizes True"
    fi
    eval $LAUNCH llava/train/train_mem.py \
        --seed $SEED \
        --model_name_or_path $VLM_PATH \
        $ALIGN_INIT \
        $AUDIO_ARGS \
        --tune_mm_mlp_adapter True \
        --data_path "\"$(to_python_list "$ALIGN_JSONL")\"" \
        --image_folder "\"$(to_python_list "$ALIGN_FOLDER")\"" \
        $COMMON_ARGS \
        --output_dir $STAGE1_OUT \
        --max_frame $ALIGN_MAX_FRAME \
        --per_device_train_batch_size $ALIGN_BATCH \
        --gradient_accumulation_steps $ALIGN_ACCUM \
        --learning_rate $ALIGN_LR \
        --save_steps 24000 \
        --model_max_length $ALIGN_MAX_LEN \
        --run_name pretrain_video_llava_$RUN_NAME
    PRETRAIN_MM_MLP_ADAPTER="$STAGE1_OUT/mm_projector.bin"
fi

# ============================================
# Stage 2: video instruction tuning (LLM + projectors; encoders frozen)
# ============================================
STAGE2_OUT="./checkpoints/video_llava_$RUN_NAME"
if [ "$STAGE" = "2" ] || [ "$STAGE" = "all" ]; then
    echo ">>> Stage 2: instruction tuning  (effective batch = $PER_DEVICE x $NUM_PROC x $ACCUM)"
    if [ -z "$PRETRAIN_MM_MLP_ADAPTER" ]; then
        echo "Error: stage 2 needs --pretrain_mm_mlp_adapter (output of stage 1: $STAGE1_OUT/mm_projector.bin)"; exit 1
    fi
    eval $LAUNCH llava/train/train_mem.py \
        --delay_load False \
        --seed $SEED \
        --model_name_or_path $VLM_PATH \
        --pretrain_mm_mlp_adapter $PRETRAIN_MM_MLP_ADAPTER \
        $AUDIO_ARGS \
        --data_path "\"$(to_python_list "$VIDEO_JSONL")\"" \
        --image_folder "\"$(to_python_list "$VIDEO_FOLDER")\"" \
        $COMMON_ARGS \
        --output_dir $STAGE2_OUT \
        --max_frame $MAX_FRAME \
        --per_device_train_batch_size $PER_DEVICE \
        --gradient_accumulation_steps $ACCUM \
        --learning_rate $LR \
        --save_steps 50000 \
        --model_max_length $MODEL_MAX_LENGTH \
        --run_name video_llava_$RUN_NAME
fi

echo "============================================"
echo "Training completed!"
[ "$STAGE" != "2" ] && echo "Stage 1 projector: $STAGE1_OUT/mm_projector.bin"
[ "$STAGE" != "1" ] && echo "Stage 2 model:     $STAGE2_OUT"
echo "============================================"
