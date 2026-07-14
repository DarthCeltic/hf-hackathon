#!/usr/bin/env bash
set -euo pipefail

repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
before="${1:?usage: refresh_trusted_smolvlm2_prs.sh BEFORE_REF [AFTER_REF] [pending-only|rerun]}"
after="${2:-HEAD}"
mode="${3:-rerun}"

if [[ "$mode" != pending-only && "$mode" != rerun ]]; then
  echo "mode must be pending-only or rerun" >&2
  exit 2
fi

before_hash="$(.github/ci/scripts/trusted_smolvlm2_input_hash.sh "$before")"
after_hash="$(.github/ci/scripts/trusted_smolvlm2_input_hash.sh "$after")"
if [[ "$before_hash" == "$after_hash" ]]; then
  echo "Trusted SmolVLM2 inputs did not change; no refresh is needed."
  exit 0
fi

count=0
while IFS=$'\t' read -r pr head_sha; do
  [[ -n "$pr" && -n "$head_sha" ]] || continue
  files="$(gh api --paginate "repos/${repo}/pulls/${pr}/files?per_page=100" --jq '.[].filename')"
  if ! grep -qxE '\.gitmodules|ported_models/llama_cpp_et/src/llama\.cpp-et' <<< "$files"; then
    continue
  fi
  gh api --method POST "repos/${repo}/statuses/${head_sha}" \
    -f state=pending \
    -f context=trusted-model/smolvlm2_500m_video \
    -f description='Main changed; trusted SmolVLM2 evaluation must run again.' \
    -f target_url="${GITHUB_SERVER_URL}/${repo}/pull/${pr}" >/dev/null
  if [[ "$mode" == rerun ]]; then
    gh workflow run trusted-smolvlm2-pr.yml --repo "$repo" --ref main -f pr_number="$pr"
    echo "Dispatched trusted SmolVLM2 evaluation for PR #$pr."
  else
    echo "Marked trusted SmolVLM2 status pending for PR #$pr."
  fi
  count=$((count + 1))
done < <(gh pr list --repo "$repo" --state open --limit 1000 --json number,headRefOid --jq '.[] | [.number, .headRefOid] | @tsv')

echo "Processed $count open trusted SmolVLM2 PR(s)."
