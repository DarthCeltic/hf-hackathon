#!/usr/bin/env bash
set -euo pipefail

ref="${1:-HEAD}"
paths=(
  .github/ci/reference/smolvlm2_500m_video.json
  .github/ci/scripts/prepare_trusted_smolvlm2_candidate.py
  .github/ci/scripts/run_smolvlm2_video_benchmark.py
  .github/ci/scripts/run_trusted_smolvlm2_candidate.sh
  .github/ci/scripts/trusted_smolvlm2_gate.py
  .github/ci/benchmark_config.json
  ported_models/llama_cpp_et/benchmarks/smolvlm2_500m_video.json
  ported_models/llama_cpp_et/artifacts.json
  ported_models/llama_cpp_et/src/llama.cpp-et
)

for path in "${paths[@]}"; do
  printf '%s\0' "$path"
  git show "$ref:$path" 2>/dev/null || true
  printf '\0'
done | sha256sum | awk '{print $1}'
