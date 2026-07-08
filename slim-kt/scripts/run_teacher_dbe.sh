#!/usr/bin/env bash
# Build the frozen-LLM teacher cache for DBE-KT22 (has real stems + MCQ options):
#   * semantic embeddings   <- MiniLM over question_text (re-encoded, no precomputed)
#   * difficulty / KCs       <- Qwen2.5-7B via local vLLM (raw question text)
#   * option ordinal labels  <- Qwen2.5-7B via local vLLM (question + choice text)
# Assumes vLLM is already serving on 127.0.0.1:8000 (start_vllm_xes.sh).
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate slimkt
export PATH=/root/miniconda3/envs/slimkt/bin:$PATH
export DATA_ROOT=/root/autodl-tmp/slim-kt/data
export OUTPUT_ROOT=/root/autodl-tmp/slim-kt/runs
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp/slim-kt
exec python -m slimkt.teacher.llm_teacher \
  --config configs/default.yaml \
  --dataset-config configs/dataset/dbe_kt22.yaml \
  --set dataset.name=dbe_kt22 teacher.llm_model=Qwen2.5-7B-Instruct
