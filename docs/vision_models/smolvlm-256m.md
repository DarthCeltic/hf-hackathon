# Porting Plan: SmolVLM-256M-Instruct

**Path:** GGUF (llama.cpp-et) — VLM (text + image → text)
**Priority:** #3 — Tiny footprint, reuses SmolLM2-135M (already ported)
**Status:** Research complete, ready for implementation

---

## Model Overview

| Property | Value |
|----------|-------|
| **Parameters** | 256M total |
| **LLM backbone** | **SmolLM2-135M-Instruct** (already ported in hackathon!) |
| **Vision encoder** | SigLIP ~93M params |
| **Projector** | MLP (2 layers, pixel-shuffle 3×3 → 9x compression) |
| **Architecture** | Idefics3-based (`idefics3` projector type) |
| **Input patches** | 512×512, 64 visual tokens/image |
| **Context** | 8,192 tokens |
| **License** | **Apache-2.0** |
| **HF Repo** | `HuggingFaceTB/SmolVLM-256M-Instruct` |

## GGUF Files (from `ggml-org/SmolVLM-256M-Instruct-GGUF`)

| File | Size |
|------|------|
| `SmolVLM-256M-Instruct-Q8_0.gguf` | 175 MB |
| `SmolVLM-256M-Instruct-F16.gguf` | 328 MB |
| `mmproj-SmolVLM-256M-Instruct-Q8_0.gguf` | 104 MB |
| `mmproj-SmolVLM-256M-Instruct-F16.gguf` | 190 MB |
| **Total (Q8_0 + mmproj Q8_0)** | **279 MB** |
| **Total (F16 + mmproj F16)** | **518 MB** |

## llama.cpp Support

- **Fully supported** since PR #13050 (merged April 22, 2025)
- Projector type: `idefics3`, `proj_scale_factor: 4`
- Pre-quantized GGUF available from `ggml-org`
- Tool: `llama-mtmd-cli`

## Benchmarks

| Benchmark | SmolVLM 256M | SmolVLM 500M | SmolVLM 2.2B |
|-----------|-------------|-------------|-------------|
| OCRBench | 52.6 | 61.0 | 65.5 |
| DocVQA Val | 58.3 | 70.5 | 79.7 |
| TextVQA Val | 49.9 | 60.5 | 72.1 |
| MMMU | 28.3 | 33.7 | 38.3 |
| ScienceQA | 73.6 | 79.7 | 84.5 |

---

## Porting Steps

### Step 1: Pin HuggingFace References

```
HF Base:      ggml-org/SmolVLM-256M-Instruct-GGUF
Revision:     <pin commit SHA from files tab>
Files:        SmolVLM-256M-Instruct-Q8_0.gguf + mmproj-SmolVLM-256M-Instruct-Q8_0.gguf
License:      apache-2.0
```

### Step 2: Add to `artifacts.json`

```json
"smolvlm_256m_q8_gguf": {
  "kind": "model",
  "framework": "llama.cpp-et",
  "variant": "SmolVLM-256M-Instruct-Q8_0",
  "filename": "SmolVLM-256M-Instruct-Q8_0.gguf",
  "env": "SMOLVLM_256M_MODEL_PATH",
  "source": {
    "type": "huggingface",
    "repo": "ggml-org/SmolVLM-256M-Instruct-GGUF",
    "revision": "<SHA>",
    "filename": "SmolVLM-256M-Instruct-Q8_0.gguf",
    "url": "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/<SHA>/SmolVLM-256M-Instruct-Q8_0.gguf"
  },
  "local_cache": "local-artifacts/models/smolvlm_256m/SmolVLM-256M-Instruct-Q8_0.gguf",
  "sha256": "<64-char-hex>",
  "size_bytes": 175000000,
  "note": "SmolVLM 256M LLM Q8_0. Uses SmolLM2-135M backbone (already ported)."
},
"smolvlm_256m_mmproj_q8": {
  "kind": "model",
  "framework": "llama.cpp-et",
  "variant": "SmolVLM-256M-mmproj-Q8_0",
  "filename": "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
  "env": "SMOLVLM_256M_MMPROJ_PATH",
  "source": {
    "type": "huggingface",
    "repo": "ggml-org/SmolVLM-256M-Instruct-GGUF",
    "revision": "<SHA>",
    "filename": "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
    "url": "..."
  },
  "local_cache": "local-artifacts/models/smolvlm_256m/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
  "sha256": "<64-char-hex>",
  "size_bytes": 104000000,
  "note": "SmolVLM 256M vision projector (SigLIP-93M + MLP). Q8_0 quantized."
}
```

### Step 3: Create Benchmark Config

File: `ported_models/llama_cpp_et/benchmarks/smolvlm_256m.json`

Key settings:
- `gpu_layers`: 99 (full offload — `idefics3` arch likely supported by ET backend since SmolLM2 is)
- `ctx_size`: 2048 (model supports 8192 but reduce for board)
- `api`: `"chat"` for VLM mode
- PPL gate: likely disabled (VLM)

### Step 4-6: Register in CI, update HF_REFERENCES.md, add recipe

Same pattern as MiniCPM-V 4.6.

---

## Infrastructure Reuse — HIGH

This is the biggest advantage of SmolVLM-256M:

- **LLM backbone IS SmolLM2-135M-Instruct** — already ported and benchmarked on the hackathon board
- The LLM decode layer is already proven on ET-SoC1
- Only the vision encoder (SigLIP ~93M params) + projector (MLP) need to be added via mmproj
- The `idefics3` projector type is well-supported in llama.cpp

## Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| 256M is very small — Q8 quantization may degrade quality | **Medium** | PR discussion noted "significantly worse" output vs Transformers at Q8. **Prefer F16 for 256M** (328 + 190 = 518 MB, still tiny) |
| `idefics3` architecture may need ET backend support | **Medium** | Check llama.cpp-et fork; SmolLM2 works, but VLM path adds vision encoder ops |
| No video support (v1) | **Low** | Use SmolVLM2-256M-Video variant if video is needed |
| VLM benchmark infrastructure not yet built | **Medium** | Shared risk across all VLM candidates; build once, use for all |

## Pre-Flight

```bash
huggingface-cli download ggml-org/SmolVLM-256M-Instruct-GGUF \
  SmolVLM-256M-Instruct-Q8_0.gguf \
  mmproj-SmolVLM-256M-Instruct-Q8_0.gguf

./llama-mtmd-cli \
  -m SmolVLM-256M-Instruct-Q8_0.gguf \
  --mmproj mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
  --image test.jpg \
  -p "What is in this image?" -n 64
```
