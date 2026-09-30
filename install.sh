#!/bin/bash
# UniMambaMia-AV
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT license
#
# Installs LLaVA + lmms-eval (as git submodules) and overlays the UniMambaMia-AV sources.
#
# Usage:
#   bash install.sh            # full install (LLaVA, lmms-eval, flash-attn, mamba-ssm, causal-conv1d, pinned deps)
#   bash install.sh --no-deps  # only copy sources and `pip install -e` without resolving dependencies

set -e

# Parse arguments
NO_DEPS=""
if [ "$1" = "--no-deps" ]; then
    NO_DEPS="--no-deps"
    echo "Installing without dependencies (--no-deps mode)"
    echo "Make sure to install dependencies manually. See README for details."
fi

# Initialize submodules (LLaVA + lmms-eval)
git submodule update --init

# ============================================
# Copy UniMambaMia-AV sources into LLaVA
# ============================================
echo "Copying UniMambaMia-AV modifications to LLaVA..."

SRC=UniMambaMia

# Core modules
cp $SRC/llava_init.py LLaVA/llava/__init__.py
cp $SRC/model_init.py LLaVA/llava/model/__init__.py
cp $SRC/llava_qwen.py LLaVA/llava/model/language_model/llava_qwen.py
cp $SRC/conversation.py LLaVA/llava/conversation.py
cp $SRC/mm_utils.py LLaVA/llava/mm_utils.py
cp $SRC/constants.py LLaVA/llava/constants.py

# Model architecture (vision tower, audio tower, audio-visual token construction)
cp $SRC/llava_arch.py LLaVA/llava/model/llava_arch.py
cp $SRC/model_builder.py LLaVA/llava/model/builder.py
cp $SRC/encoder_builder.py LLaVA/llava/model/multimodal_encoder/builder.py
cp $SRC/clip_encoder.py LLaVA/llava/model/multimodal_encoder/clip_encoder.py

# Projectors: vision projector + UniMambaMia audio compressor (MambaMia2 backbone)
cp $SRC/projector_builder.py LLaVA/llava/model/multimodal_projector/builder.py
cp $SRC/audio_compressor.py LLaVA/llava/model/multimodal_projector/audio_compressor.py
cp $SRC/configuration_mambamia2.py LLaVA/llava/model/multimodal_projector/configuration_mambamia2.py
cp $SRC/modeling_mambamia2.py LLaVA/llava/model/multimodal_projector/modeling_mambamia2.py

# Video/audio loading utilities
cp $SRC/video_utils.py LLaVA/llava/video_utils.py

# Training modules
cp $SRC/train.py LLaVA/llava/train/train.py
cp $SRC/train_mem.py LLaVA/llava/train/train_mem.py
cp $SRC/llava_trainer.py LLaVA/llava/train/llava_trainer.py

# Project configuration (custom dependencies)
cp $SRC/pyproject.toml LLaVA/pyproject.toml

echo "LLaVA + UniMambaMia-AV setup complete!"

# ============================================
# lmms-eval: model wrapper + audio-visual benchmarks
# ============================================
echo "Setting up lmms-eval..."
cp $SRC/unimambamia_vid.py lmms-eval/lmms_eval/models/

# Register the model in AVAILABLE_MODELS dict if not already registered
if ! grep -q "unimambamia_vid" lmms-eval/lmms_eval/models/__init__.py; then
    # Add to AVAILABLE_MODELS dictionary (before the closing brace)
    sed -i '/"aria": "Aria",/a\    "unimambamia_vid": "UniMambaMiaVid",' lmms-eval/lmms_eval/models/__init__.py
    echo "Added UniMambaMiaVid to lmms-eval AVAILABLE_MODELS"
fi

# GPT-4o single-frame probe used for the benchmark audit (register name: gpt4o)
cp $SRC/gpt4o_single_frame.py lmms-eval/lmms_eval/models/gpt4o_single_frame.py
if ! grep -q '"gpt4o":' lmms-eval/lmms_eval/models/__init__.py; then
    sed -i '/"aria": "Aria",/a\    "gpt4o": "GPT4O",' lmms-eval/lmms_eval/models/__init__.py
    echo "Added GPT4O (single-frame probe) to lmms-eval AVAILABLE_MODELS"
fi

# Joins the official AVQA / Music-AVQA annotations onto the released item ids at load time
cp $SRC/official_annotations.py lmms-eval/lmms_eval/tasks/_task_utils/official_annotations.py

# Benchmarks used in the paper that are not part of the pinned lmms-eval:
#   avqa_2025, avqa_hard, music_avqa, music_avqa_hard, av_speakerbench, worldsense, videommmu
for task in $SRC/lmms_eval_tasks/*/; do
    task_name=$(basename "$task")
    echo "  adding lmms-eval task: $task_name"
    rm -rf "lmms-eval/lmms_eval/tasks/$task_name"
    cp -r "$task" "lmms-eval/lmms_eval/tasks/$task_name"
done

# ActivityNet-QA is scored by a GPT judge; the pinned lmms-eval still points at a retired model.
sed -i 's/gpt_eval_model_name: gpt-3.5-turbo-0613/gpt_eval_model_name: gpt-3.5-turbo-0125/' \
    lmms-eval/lmms_eval/tasks/activitynetqa/_default_template_yaml

# Install LLaVA and lmms-eval
echo ""

# flash-attn
echo "Installing flash-attn (pre-built wheel)..."
# Pre-built wheel for Python 3.10, PyTorch 2.4, CUDA 12.1
pip install $NO_DEPS https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# mamba_ssm
if python -c "import mamba_ssm" 2>/dev/null; then
    echo "mamba_ssm already installed, skipping..."
else
    echo "Installing mamba_ssm (pre-built wheel)..."
    # Pre-built wheel for Python 3.10, PyTorch 2.4, CUDA 12.1
    pip install $NO_DEPS https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi

# causal-conv1d
echo ""
if python -c "import causal_conv1d" 2>/dev/null; then
    echo "causal-conv1d already installed, skipping..."
else
    echo "Installing causal-conv1d (pre-built wheel)..."
    # Pre-built wheel for Python 3.10, PyTorch 2.4, CUDA 12.1
    pip install $NO_DEPS https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.4/causal_conv1d-1.5.4+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi

echo ""
echo "Installing LLaVA..."
pip install $NO_DEPS -e LLaVA

echo ""
echo "Installing lmms-eval..."
pip install $NO_DEPS -e lmms-eval

# nltk
python - <<END
import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('punkt')
END

# tested env
pip install transformers==4.50.0 pyarrow==19.0.0 numpy==1.26.4 librosa soundfile

# Patch transformers GenerationConfig validation (too strict in 4.50.0)
echo "Patching transformers GenerationConfig..."
TRANSFORMERS_PATH=$(python -c "import transformers; print(transformers.__path__[0])")
cp $SRC/configuration_utils.py "$TRANSFORMERS_PATH/generation/configuration_utils.py"

echo ""
echo "============================================"
echo "UniMambaMia-AV installation complete!"
echo ""
if [ -n "$NO_DEPS" ]; then
    echo "Installed in --no-deps mode."
    echo "Please install dependencies manually (see README)."
else
    echo "LLaVA and lmms-eval installed with dependencies."
fi
echo "============================================"
