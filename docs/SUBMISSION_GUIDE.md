# Submission Guide

Submit through GitHub PRs against:

<https://github.com/aifoundry-org/hf-hackathon>

The Hugging Face repo is a read-only mirror. Do not submit hackathon PRs there.

Need help? Ask in Discord `#Lab`: <https://discord.gg/CbSA2umxf6>

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
5. Do not edit `data/*.json` or the generated leaderboard block in `README.md`;
   board CI updates those after merge.
6. Do not commit model blobs, local outputs, secrets, or machine-specific paths.
7. Run:

   ```bash
   bash .github/ci/scripts/ci_preflight.sh
   ```

8. Open the GitHub PR.

PR board CI selects the configured models touched by the diff and comments with
the ET-SoC1 result. Public fork PRs do not run directly on the board host; a
maintainer can move trusted changes to a branch that can run board CI.
