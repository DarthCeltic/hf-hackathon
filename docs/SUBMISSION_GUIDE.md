# Submission Guide

This guide is for the AIFoundry + OpenHW CORE-ET hackathon. Board CI currently
runs submissions on ET-SoC1 boards and comments back with reproducible results.

Submit through GitHub PRs against:

<https://github.com/aifoundry-org/hf-hackathon>

The Hugging Face repo is a read-only mirror. Do not submit hackathon PRs there.

Need help? Ask in Discord `#community-lab`: <https://discord.gg/CbSA2umxf6>

## PR Checklist

1. Add or update a folder under `ported_models/`.
2. For an existing framework, update its `artifacts.json`, benchmark JSON, and
   `.github/ci/benchmark_config.json` entry as needed.
3. For a new framework, commit the framework source under
   `ported_models/<name>/src/`, add `artifacts.json`, add a runner under
   `.github/ci/scripts/`, and register it in the benchmark config/helper.
4. Use a Hugging Face model repo as the base reference whenever the model family
   exists there. Pin the repo, revision, filename(s), and license; document any
   export, quantization, preprocessing, shape, or packing step needed to
   reproduce the ET-SoC1 artifact.
5. Satisfy the fully validated model standard below for every leaderboard model
   touched by the PR.
6. Add a reusable `.md` recipe or equivalent agent-readable notes. Include the
   task breakdown, instructions or prompt files used, repos/docs/RTL/model files
   you pointed tools at, relevant commands, environment assumptions,
   verification steps, and failures or dead ends that would help another
   participant or agent reproduce the path. A `SKILLS.md`-style file is fine,
   but not required.
7. Do not edit `data/*.json` or the generated leaderboard block in `README.md`;
   board CI updates those after merge.
8. Do not commit model blobs, local outputs, secrets, or machine-specific paths.
9. Run:

   ```bash
   bash .github/ci/scripts/ci_preflight.sh
   ```

10. Open the GitHub PR.

## Fully Validated Model Standard

A leaderboard submission should prove that the ET-SoC1 board ran the intended
model, not just a fast kernel with the same name. A valid model submission needs
all five pieces below.

1. **Provenance.** Declare the upstream reference: Hugging Face repo, pinned
   commit SHA, exact file names, file hashes, license, and the conversion,
   export, quantization, preprocessing, or packing commands used to produce the
   ET-SoC1 artifact. Do not use floating revisions such as `main` for the model
   identity. If a workload uses synthetic weights, label it as a kernel
   benchmark rather than a validated model row.
2. **Host Reference.** Provide a host-side oracle for each benchmark case. The
   oracle should run the pinned upstream model or the pinned exported artifact
   with the same preprocessing and quantization path used by the board run. The
   input, weights or model artifact, and expected output must be deterministic
   and reproducible from the documented commands.
3. **Board Correctness.** Compare ET-SoC1 output against the host reference in
   CI. Tensor/image models should report bounded error metrics such as
   `max_abs`, `mean_abs`, or `rmse`, and should reject degenerate outputs such
   as all-zero or constant tensors. Detection models should check expected
   classes, scores, and boxes. Language models should include a quality metric
   such as perplexity.
4. **Model Quality.** Report a task-level quality signal in addition to speed:
   for example PPL for language models, expected detections for object
   detection, PSNR/SSIM or denoising improvement for image restoration, or
   another task-appropriate metric documented in the model card. A faster result
   that materially degrades quality is not a valid leaderboard improvement.
5. **Board Performance and Leaderboard Update.** Report the board performance
   metric used for ranking, the board configuration, commit SHA, run URL, and
   participant attribution. CI must produce the score artifact, the leaderboard
   gate must pass, and the main-branch board workflow must update `data/*.json`
   after merge.

PR board CI selects the configured models touched by the diff and comments with
the ET-SoC1 board result. After merge, the main-branch board run uses the same
touched-model selection when updating leaderboard data. External fork PRs reach
the board runner only after GitHub's external-contributor workflow approval gate
releases the run.

The `Leaderboard gate` check is the merge signal for benchmarked submissions.
Every selected model must produce a passing board score and strictly improve the
current base-branch leaderboard value for that model's primary metric. Higher is
better for token/second metrics; lower is better for kernel wait time. A new
configured model with no prior leaderboard entry can pass by producing its first
valid board score. Set the optional repository variable
`LEADERBOARD_MIN_RELATIVE_IMPROVEMENT` to require a larger fractional gain, such
as `0.01` for 1%.

CI/scoring-only changes that do not change submitted model code or
runtime-affecting model config still need passing board scores for the selected
models, but they do not need to improve the leaderboard runtime.

For ELF benchmark models, a passing board score also includes the configured
dump accuracy gate in `.github/ci/benchmark_config.json`. The current gates
check YOLO against host-generated detections from the pinned reference model on
five public, hash-pinned COCO images. Correctness is an eligibility gate; only a
passing result is compared by mean end-to-end latency.

For models that run `llama-perplexity`, the same gate also protects quality:
the PR score must include PPL, and it must be no more than 20% worse than the
best PPL currently recorded for that model. Set
`LEADERBOARD_MAX_PPL_REGRESSION` to adjust that fractional limit.

After a PR is merged, the main-branch board run credits leaderboard entries to
the merged commit's author. If GitHub can map that author to a user, the
leaderboard uses the GitHub login; otherwise it falls back to the raw git author
name.
