# Third-Party Components

The first-party code in this repository is licensed under **Apache-2.0** (see
[`LICENSE`](../LICENSE) and [`NOTICE`](../NOTICE)). The vendored Esperanto
sys-emu firmware under `.github/ci/firmware/esperanto-fw/` is also owned by us
and distributed under Apache-2.0. The components below are **not** covered by
that license; each retains its own upstream license and copyright. This file is
the top-level inventory; per-port `THIRD_PARTY.md` files hold the detailed
records.

| Component | Path | Type | License | Notes |
|-----------|------|------|---------|-------|
| ET `llama.cpp` fork | `ported_models/llama_cpp_et/src/llama.cpp-et` | git submodule (pointer) | MIT | Upstream `aifoundry-org/llama.cpp`, branch `et`. License travels with the submodule. See `ported_models/llama_cpp_et/THIRD_PARTY.md`. |
| GGONNX | `ported_models/ggonnx/src/ggonnx` | vendored source | Pending (expected Apache-2.0) | Upstream `marty1885/ggonnx` had no LICENSE at vendoring time; license grant being secured with the author. See `ported_models/ggonnx/THIRD_PARTY.md`. |
| Model weights (GGUF, ONNX) | not committed | downloaded at runtime | Per upstream model card | Fetched on the board host from Hugging Face / source URLs declared in each port's `artifacts.json`; each model retains its own license. |

## How licensing is structured here

- `LICENSE` (Apache-2.0) applies to this repository's **own** code: the CI
  scripts, configs, docs, porting harness, and vendored Esperanto sys-emu
  firmware.
- `NOTICE` carries the Apache attribution required for redistribution.
- Bundled third-party source/binaries keep their upstream license; do not assume
  Apache-2.0 covers anything listed above.
- Each `ported_models/<port>/THIRD_PARTY.md` records the upstream URL, pinned
  revision, vendoring date, and license status for that port.

If you add a port that vendors or submodules external code, record it here and in
your port's `THIRD_PARTY.md` (see [`SUBMISSION_GUIDE.md`](SUBMISSION_GUIDE.md)).
