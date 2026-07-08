#!/usr/bin/env bash
# Start the Qwen2.5-7B vLLM OpenAI-compatible server for the SLIM-KT teacher.
# Loads the local model (no network), puts the conda env bin on PATH so ninja is
# found, and runs eager (no torch.compile) for robust startup on fresh instances.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate slimkt
export PATH=/root/miniconda3/envs/slimkt/bin:$PATH
cd /root/autodl-tmp/slim-kt
exec python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --port 8000 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --enforce-eager
