# SmolVLM-256M-Instruct Q8_0 — llama.cpp-et board benchmark

## First VLM in the hackathon

SmolVLM-256M-Instruct is the first vision-language model (VLM) ported to the
ET-SoC1 board. It uses the Idefics3 architecture with a SigLIP vision encoder
(~93M params) and an MLP projector that bridges image embeddings to the LLM
backbone. The LLM backbone is SmolLM2-135M-Instruct, already proven on ET-SoC1
as `smollm2_135m_q8_gguf`.

The vision projector (`mmproj`) is a new addition requiring the `--mmproj` flag
in llama-server. This enables image-to-text inference entirely on the ET-SoC1.

## Hugging Face base

| Field | Value |
|-------|-------|
| Repo | [ggml-org/SmolVLM-256M-Instruct-GGUF](https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF) |
| Revision | `b9e4379657e1450d04d02eec8e345667265b0a00` |
| LLM file | `SmolVLM-256M-Instruct-Q8_0.gguf` (175 MiB) |
| mmproj file | `mmproj-SmolVLM-256M-Instruct-Q8_0.gguf` (104 MiB) |
| License | Apache-2.0 |
| Export step | None (upstream Q8_0 GGUF used as-is) |

## Architecture

- **Architecture family**: Idefics3 (projector type `idefics3`)
- **Vision encoder**: SigLIP ~93M parameters
- **LLM backbone**: SmolLM2-135M-Instruct (HuggingFace transformer decoder)
- **Projector**: MLP bridging SigLIP vision tokens to LLM embedding space
- **llama.cpp support**: PR [#13050](https://github.com/ggerganov/llama.cpp/pull/13050) merged April 22, 2025

## ET backend settings

Mirrors the `smollm2_135m` row (same LLM backbone):
`device=ET`, `gpu_layers=99`, completion API, `ctx_size=2048`.

| Parameter | Value |
|-----------|-------|
| device | ET |
| gpu_layers | 99 |
| ctx_size | 2048 |
| batch_size | 256 |
| ubatch_size | 128 |
| port | 18100 |
| ready_timeout_s | 120 |
| request_timeout_s | 240 |

## Benchmark notes

The benchmark exercises text-only generation (completion API) and WikiText-2
perplexity, same as the SmolLM2-135M row. Since the LLM backbone is identical,
PPL and decode throughput should closely match `smollm2_135m`. Vision inference
via the mmproj projector is not benchmarked by the standard text harness but is
available when `--mmproj` is passed to llama-server.

## Files added/changed

- `ported_models/llama_cpp_et/artifacts.json` — `smolvlm_256m_q8_gguf` and `smolvlm_256m_mmproj_q8` artifacts
- `ported_models/llama_cpp_et/benchmarks/smolvlm_256m.json` — board runner config
- `.github/ci/benchmark_config.json` — `smolvlm_256m` model key
- `docs/HF_REFERENCES.md` — HuggingFace reference row

## Verification

```bash
bash .github/ci/scripts/ci_preflight.sh
python .github/ci/scripts/benchmark_config_helpers.py --target board --models smolvlm_256m --format space
```

Board CI runs decode tokens/s and WikiText-2 raw PPL via `run_llamaserver_benchmark.py`.

## References

- [SUBMISSION_GUIDE.md](../../../docs/SUBMISSION_GUIDE.md)
- [HF_REFERENCES.md](../../../docs/HF_REFERENCES.md)
- Same LLM backbone: `benchmarks/smollm2_135m.json`
- llama.cpp Idefics3 support: PR #13050
