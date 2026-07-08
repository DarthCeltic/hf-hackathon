# SmolVLM-500M-Instruct Q8_0 — llama.cpp-et board benchmark

## The sweet spot VLM

SmolVLM-500M-Instruct sits between the 256M and 2.2B variants, offering a
strong balance of capability and efficiency. It uses the Idefics3 architecture
with a SigLIP vision encoder (~93M params, same as the 256M variant) and an MLP
projector with pixel-shuffle 3×3 that bridges image embeddings to the LLM
backbone. The LLM backbone is SmolLM2-360M-Instruct, already proven on ET-SoC1
as `smollm2_360m_q8_gguf`.

## Hugging Face base

| Field | Value |
|-------|-------|
| Repo | [ggml-org/SmolVLM-500M-Instruct-GGUF](https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF) |
| Revision | `72e986006ef53e37cdd3f6d4241c90b0f01df376` |
| LLM file | `SmolVLM-500M-Instruct-Q8_0.gguf` (~417 MiB / 436,806,912 bytes) |
| mmproj file | `mmproj-SmolVLM-500M-Instruct-Q8_0.gguf` (~104 MiB / 108,783,360 bytes) |
| Total Q8_0 | ~521 MiB (545,590,272 bytes) |
| License | Apache-2.0 |
| Export step | None (upstream Q8_0 GGUF used as-is) |

## Architecture

- **Architecture family**: Idefics3 (projector type `idefics3`)
- **Vision encoder**: SigLIP ~93M parameters (same encoder as SmolVLM-256M)
- **LLM backbone**: SmolLM2-360M-Instruct (HuggingFace transformer decoder, already ported)
- **Projector**: MLP (2 layers) with pixel-shuffle 3×3, bridging SigLIP vision tokens to LLM embedding space
- **llama.cpp support**: PR [#13050](https://github.com/ggerganov/llama.cpp/pull/13050) merged April 22, 2025

## ET backend settings

Mirrors the `smollm2_360m` row (same LLM backbone):
`device=ET`, `gpu_layers=99`, completion API, `ctx_size=2048`.

| Parameter | Value |
|-----------|-------|
| device | ET |
| gpu_layers | 99 |
| ctx_size | 2048 |
| batch_size | 256 |
| ubatch_size | 128 |
| port | 18103 |
| ready_timeout_s | 120 |
| request_timeout_s | 240 |

## Total deployment size

At Q8_0 quantization the full VLM stack (LLM + vision projector) weighs
approximately 521 MiB — well within the ET-SoC1's memory budget and a
comfortable fit between the 256M variant (~266 MiB total) and the 2.2B variant.

## Benchmarks

SmolVLM-500M achieves strong multimodal benchmarks:

| Benchmark | Score |
|-----------|-------|
| OCRBench | 61.0 |
| DocVQA | 70.5 |
| TextVQA | 69.5 |
| MMMU | 34.1 |

These place it as a capable vision-language model for document understanding,
OCR, and visual question answering at a fraction of the cost of larger VLMs.

## Infrastructure reuse

This port reuses infrastructure already established for the SmolLM2-360M row:
- Same llama-server binary and ET backend
- Same WikiText-2 perplexity harness
- Same board deployment workflow
- Vision projector loaded via `--mmproj` flag in llama-server

## Files added/changed

- `ported_models/llama_cpp_et/artifacts.json` — `smolvlm_500m_q8_gguf` and `smolvlm_500m_mmproj_q8` artifacts
- `ported_models/llama_cpp_et/benchmarks/smolvlm_500m.json` — board runner config
- `.github/ci/benchmark_config.json` — `smolvlm_500m` model key
- `docs/HF_REFERENCES.md` — HuggingFace reference row

## Verification

```bash
bash .github/ci/scripts/ci_preflight.sh
python .github/ci/scripts/benchmark_config_helpers.py --target board --models smolvlm_500m --format space
```

Board CI runs decode tokens/s and WikiText-2 raw PPL via `run_llamaserver_benchmark.py`.

## References

- [SUBMISSION_GUIDE.md](../../../docs/SUBMISSION_GUIDE.md)
- [HF_REFERENCES.md](../../../docs/HF_REFERENCES.md)
- Same LLM backbone: `benchmarks/smollm2_360m.json`
- Smaller variant: `benchmarks/smolvlm_256m.json`
- llama.cpp Idefics3 support: PR #13050
