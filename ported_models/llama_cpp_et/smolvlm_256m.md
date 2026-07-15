## Overview
This recipe documents the addition of the `smolvlm_256m` model in GGUF format to the `llama.cpp-et` framework for the AIFoundry CORE-ET Hackathon.

## Model Reference
- **Hugging Face Repository**: `ggml-org/SmolVLM-256M-Instruct-GGUF`
- **Revision**: `b9e4379657e1450d04d02eec8e345667265b0a00`
- **Filename**: `SmolVLM-256M-Instruct-Q8_0.gguf`
- **Format**: Q8_0 GGUF

## Steps Taken
1. **Steps**: As smolvlm_256m is already present in the ported models artifacts, just ran the preflight and soc3 benchmark. Detailed steps below. Ran Text only per existing benchmark.

## Steps to reproduce
1. From the hackathon root repo .github/ci/platform/deploy/install-soc3-ssh-key.sh
2. MODELS="smolvlm_256m"   .github/ci/platform/deploy/soc3-benchmark.sh