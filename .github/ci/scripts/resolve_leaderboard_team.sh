#!/usr/bin/env bash
set -euo pipefail

sha="${1:-${GITHUB_SHA:-HEAD}}"
team="${GITHUB_ACTOR:-local}"

if [[ "${GITHUB_EVENT_NAME:-}" == "push" && "${GITHUB_REF:-}" == "refs/heads/main" ]]; then
  author_sha="$sha"
  parents="$(git show -s --format=%P "$sha" 2>/dev/null || true)"
  read -r -a parent_shas <<< "$parents"
  if (( ${#parent_shas[@]} >= 2 )); then
    author_sha="${parent_shas[1]}"
  fi

  author="$(git show -s --format=%an "$author_sha" 2>/dev/null || true)"
  if [[ -n "$author" ]]; then
    team="$author"
  fi
fi

printf '%s\n' "$team"
