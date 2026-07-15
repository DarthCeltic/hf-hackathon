# YOLO FP32 Tensor/TFMA Optimization

This recipe describes the ET-SoC1 optimization in `yolo_tensor.h`. It keeps
the existing FP32 model, preprocessing, postprocessing, weight blob, and
five-image correctness contract unchanged. Only the 1x1 and stride-1 padded
3x3 convolution implementations change.

## Result

The compatible `main` leaderboard baseline was Rehan Qasim's merged YOLO PR
#102 at `1.484214 s` mean latency. A 30% reduction therefore required no more
than `1.0389498 s`.

The clean candidate measured `0.9175406 s` across the five fixed COCO cases:

| Case | Kernel wait |
| --- | ---: |
| `coco_room_000139` | 0.917314 s |
| `coco_cat_524280` | 0.917170 s |
| `coco_giraffes_296969` | 0.918133 s |
| `coco_elephants_445248` | 0.917615 s |
| `coco_baseball_043816` | 0.917471 s |
| Mean | **0.9175406 s** |

This is 38.18% lower latency than the compatible baseline (1.618x speedup)
and is 121.4 ms below the 30%-reduction threshold. Current-main scoring passed
all five cases with 18/18 matched detections, precision and recall of 1.0,
mean IoU `0.99999915`, confidence MAE `3.06e-7`, and validation contract
`976947a7f36e6db78c58e20d2220acc539ddd1335c07e11ffbacee7e3403dfc5`.

These numbers are development-board measurements. The trusted GitHub gate is
the authoritative submission result.

## Material Reviewed

The optimization work used these sources:

- `docs/opinionated_porting_options/afonso.md` for staged PMC measurement and
  cache-counter interpretation;
- `docs/opinionated_porting_options/martin.md` for ET-SoC1 pipeline, cache, and
  VPU guidance;
- `ported_models/yolo/src/yolo_common.h` and `yolo_m30_argbuf.c` for the
  existing layouts, barriers, repacking, and cache ownership;
- `et-platform/sw-sysemu/insns/tensors.cpp` and
  `et-common-libs/include/erbium*/isa/tensors.h` for the exact tensor-load,
  TFMA, TenB, and tensor-store semantics;
- the validated FP32/FP16/INT8 TFMA probes and notes in the internal
  `nekkoai/erbium-hackathon` optimization repository;
- the current YOLO leaderboard and trusted result for PR #102;
- earlier YOLO tensor experiments in PRs #72 and #101, especially their
  accuracy failures and per-tap packing/cache overhead.

No benchmark contract, fixture, host oracle, scorer, runner, or leaderboard
JSON is changed by this optimization.

## Profiling Method

Temporary instrumentation divided the model into preprocess, backbone blocks,
SPPF, PSA, FPN, heads, DFL, and postprocess stages. Each active T0 hart sampled
`hpmcounter3` through `hpmcounter8`; T0 also sampled shire-cache and memory-shire
counters through the supported syscalls. A second temporary ledger timed every
1x1, stride-1 3x3, and stride-2 3x3 call.

The stage profile attributed about 95% of model cycles to convolution. The
whole-run shire-cache counters reported roughly 54.5M reads and 6.5M writes,
while the memory-shire counters reported only about 1.35M reads and 1.14M
writes. The board was compute/pipeline limited rather than DRAM-bandwidth
limited.

The call ledger at the board's 600 MHz minion clock was:

| Kernel family | Before | After |
| --- | ---: | ---: |
| 1x1, 41 calls | 0.354247 s | 0.081796 s |
| stride-1 3x3, 24 calls | 0.802623 s | 0.529536 s |
| stride-2 3x3, 4 calls | 0.221045 s | 0.220209 s |

All PMC regions, checkpoints, call timers, and profiling defines were removed
before the clean build and five-case validation.

## Final Kernel Mapping

### 1x1

NCHW already provides a tensor-friendly matrix multiplication. For an OC16 x
HW16 output tile, normal tensor loads read a 16x16 weight tile from
`[OC][IC]` and a 16x16 activation tile from `[IC][HW]`. FP32 TFMA accumulates
over IC16 tiles, tensor-store writes FP32 results, and the existing vector SiLU
epilogue adds bias and activation. No input or weight repack is needed.

### Stride-1 padded 3x3

The implementation repacks weights once per call as
`[OC16 tile][ky,kx][IC][OC lane]`. A `transpose32` tensor load converts 16 IC
cache lines into the required `[OC lane][IC lane]` A matrix.

For spatial tiles aligned to 16 output columns, the center `kx=1` input column
is also cache-line aligned. TFMA computes all three vertical taps for that
column without im2col. The board-proven OC4 VPU pipeline then adds the shifted
`kx=0` and `kx=2` columns, bias, and SiLU. This tensors one third of the 3x3
MACs without paying for a full im2col buffer.

Work is partitioned over `(OC16 tile, output row, output-width tile)` rather
than only OC16. That detail is important: OC-only partitioning left most of
the eight T0 minions idle on the common 16/32/64-channel layers.

Tensor operations overwrite the floating-point register file. The kernel
therefore declares all FREGs clobbered after every tensor store before scalar
or VPU floating-point code resumes. It also invalidates destination lines
before tensor stores and evicts completed output lines because the ET-SoC1 L1
cache is non-coherent in scratchpad mode.

## Experiments That Did Not Help

- A wider OC4 spatial-x4 VPU kernel stayed accurate but slowed the room case to
  `1.63836 s`; register pressure and pipeline/I-cache effects outweighed the
  instruction reduction.
- Adding `noinline` to hot 3x3 kernels appeared to reach `1.38702 s`, but it
  produced 54 bogus detections and was rejected.
- A spatial-x2 3x3 branch slowed to `1.59349 s` and corrupted detections.
- The first correct tensor 3x3 implementation partitioned only by OC16. It
  slowed the candidate to `1.65073 s` because too few minions received work.
- Full per-tap im2col/copy/evict designs from earlier tensor attempts had too
  much data-marshalling overhead and had already failed the five-image gate.

## Reproduction

Follow `docs/ET_SOC1_QUICKSTART.md` to install the ET toolchain and prepare the
fixed YOLO assets. From the repository root, build the board ELF with:

```bash
export BOARD_BENCHMARK=1
export BENCHMARK_DEVICE=soc1sim
export REPO_ROOT="$PWD"
export BENCHMARK_ARTIFACT_ROOT="${BENCHMARK_ARTIFACT_ROOT:-$PWD/local-artifacts/model-port-benchmarks}"

.github/ci/scripts/prepare_benchmark_inputs.sh yolo
.github/ci/scripts/build_leaderboard_elf.sh yolo
```

Run the standard board workflow rather than a private launcher wrapper so it
generates the host reference, executes all five cases, and invokes the same
scorer as CI:

```bash
export SOC3_HOST="root@${BOARD_HOST}"
export MODELS=yolo
.github/ci/platform/deploy/soc3-benchmark.sh
```

Finally run the repository checks:

```bash
bash .github/ci/scripts/ci_preflight.sh
git diff --check
```

Do not commit generated ELFs, dumps, score JSON, raw profiling regions, board
logs, or edits to `data/yolo.json`; trusted board CI owns the score and
leaderboard update.
