#!/usr/bin/env bash
set -euo pipefail

sha="${1:-${GITHUB_SHA:-HEAD}}"
team="${GITHUB_ACTOR:-local}"

if [[ "${GITHUB_EVENT_NAME:-}" == "push" && "${GITHUB_REF:-}" == "refs/heads/main" ]]; then
  login=""
  api_token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  if [[ -n "${GITHUB_REPOSITORY:-}" && -n "$api_token" ]] && command -v gh >/dev/null 2>&1; then
    login="$(GH_TOKEN="$api_token" gh api "repos/${GITHUB_REPOSITORY}/commits/${sha}" --jq '.author.login // empty' 2>/dev/null || true)"
  fi

  if [[ -n "$login" ]]; then
    team="$login"
  else
    author="$(git show -s --format=%an "$sha" 2>/dev/null || true)"
    if [[ -n "$author" ]]; then
      team="$author"
    fi
  fi
fi

printf '%s\n' "$team"
