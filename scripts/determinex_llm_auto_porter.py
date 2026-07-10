#!/usr/bin/env python3
"""Determinex LLM Auto-Porter for ET-SoC1.

HARDENED 2026-07-10 after a live audit found 0/10 of this script's prior
output survived manual review (unsafe/unofficial source, unverifiable model
family, embedding/VLM registered on a text-completion harness, a duplicate
base model, and one literal HF "repo moved" redirect page treated as a
model). Root cause: this script never routed through any Determinex
verification module (no oracle, no safety gate) -- it was a bare HF scraper
sharing the product name. See corpus/programbench/build_knowledge.json
"auto_porter_quality_gaps_2026_07_10" for the full audit.

This version adds four gates, all checked BEFORE a candidate is written:
1. Publisher allowlist -- only orgs with a track record of correctly-labeled
   quantizations. Unknown publishers are SKIPPED, not silently trusted.
2. pipeline_tag task-type check -- `api.model_info()` already returns this;
   only "text-generation" (causal LM) is accepted. Embedding / VLM / other
   task types are skipped, not force-fit into the text-completion harness.
3. Base-model dedup -- normalizes the repo name to a base-model identity
   (strips quantizer/org noise) and skips if that identity is already
   registered under ANY existing artifact key, not just an exact ID match.
4. Junk-repo filter -- rejects known non-model HF redirect/placeholder repo
   name patterns.

Every skip is printed with its reason so a human can audit what was
excluded and why -- silence is not a verification strategy.
"""
import argparse
import json
import os
import re
from huggingface_hub import HfApi

# Gate 1: publisher allowlist. Extend deliberately, not by dumping in every
# org that shows up -- each addition should be a publisher whose model cards
# and naming have been manually spot-checked as accurate.
TRUSTED_PUBLISHERS = {
    "ggml-org", "unsloth", "hugging-quants", "lmstudio-community",
    "bartowski", "TheBloke", "Qwen", "meta-llama", "google",
    "microsoft", "mistralai",
}

# Gate 4: junk-repo name patterns that are never real models.
JUNK_REPO_PATTERNS = (
    re.compile(r"models?[-_]?moved", re.I),
    re.compile(r"repo[-_]?moved", re.I),
    re.compile(r"deprecated", re.I),
    re.compile(r"^test[-_]", re.I),
    re.compile(r"placeholder", re.I),
)

# Gate 2: only these HF pipeline_tags get the causal-LM text-completion
# harness. Everything else needs a different (currently unimplemented)
# harness and must not be force-fit into this one.
ACCEPTED_PIPELINE_TAGS = {"text-generation", "text2text-generation"}


def base_model_identity(repo_id: str, filename: str) -> str:
    """Normalize a repo+filename to a coarse base-model identity for dedup,
    stripping quantizer-org noise and quant-suffix noise. Two different
    repackagings of the same base weights should collide here even if their
    literal artifact_id strings differ."""
    name = repo_id.split("/")[-1]
    name = re.sub(r"[-_](gguf|GGUF)$", "", name)
    name = re.sub(r"[-_](q4|q5|q6|q8)[-_]?[a-z0-9]*$", "", name, flags=re.I)
    name = re.sub(r"[^a-z0-9]+", "", name.lower())
    return name


def is_junk_repo(repo_id: str) -> bool:
    name = repo_id.split("/")[-1]
    return any(p.search(name) for p in JUNK_REPO_PATTERNS)


def main():
    parser = argparse.ArgumentParser(description="Determinex LLM Auto-Porter for ET-SoC1")
    parser.add_argument("--tags", nargs='+', default=["gguf", "q8_0"], help="Tags to filter models")
    parser.add_argument("--max-size", type=int, default=3000000000, help="Max model size in bytes (3GB)")
    parser.add_argument("--limit", type=int, default=10, help="Number of models to process")
    parser.add_argument("--artifacts", type=str, default="ported_models/llama_cpp_et/artifacts.json", help="Path to artifacts.json")
    parser.add_argument("--benchmarks", type=str, default="ported_models/llama_cpp_et/benchmarks", help="Path to benchmarks dir")
    parser.add_argument("--allow-publisher", action="append", default=[],
                         help="Add a publisher to the trusted allowlist for this run (repeatable).")
    args = parser.parse_args()

    trusted = TRUSTED_PUBLISHERS | set(args.allow_publisher)

    api = HfApi()
    models = api.list_models(filter="gguf", sort="downloads", limit=100)

    with open(args.artifacts, "r") as f:
        artifacts_data = json.load(f)

    existing_identities = set()
    for key, entry in artifacts_data.get("artifacts", {}).items():
        src = entry.get("source", {})
        if src.get("type") == "huggingface" and src.get("repo"):
            existing_identities.add(base_model_identity(src["repo"], entry.get("filename", "")))

    count = 0
    skipped = {"publisher": 0, "junk_repo": 0, "task_type": 0, "duplicate": 0, "size": 0, "error": 0}
    for m in models:
        if count >= args.limit:
            break
        repo_id = m.id
        publisher = repo_id.split("/")[0] if "/" in repo_id else ""
        print(f"Checking {repo_id}...")

        if is_junk_repo(repo_id):
            print(f"  SKIP (junk-repo pattern): {repo_id}")
            skipped["junk_repo"] += 1
            continue

        if publisher not in trusted:
            print(f"  SKIP (untrusted publisher '{publisher}'): {repo_id}")
            skipped["publisher"] += 1
            continue

        try:
            info = api.model_info(repo_id, files_metadata=True)
        except Exception as e:
            print(f"  SKIP (error fetching info): {repo_id}: {e}")
            skipped["error"] += 1
            continue

        pipeline_tag = getattr(info, "pipeline_tag", None)
        if pipeline_tag not in ACCEPTED_PIPELINE_TAGS:
            print(f"  SKIP (pipeline_tag='{pipeline_tag}', not a causal LM): {repo_id}")
            skipped["task_type"] += 1
            continue

        for file in info.siblings:
            filename = getattr(file, "rfilename", "")
            size = getattr(file, "size", 0)
            if not filename.endswith(".gguf"):
                continue

            tag_match = True
            for tag in args.tags:
                if tag != "gguf" and tag.lower() not in filename.lower():
                    tag_match = False
                    break
            if not tag_match:
                continue

            if size > args.max_size:
                print(f"  Skipping {filename}: size {size} > {args.max_size}")
                skipped["size"] += 1
                continue

            identity = base_model_identity(repo_id, filename)
            if identity in existing_identities:
                print(f"  SKIP (duplicate base model '{identity}', already registered): {repo_id}")
                skipped["duplicate"] += 1
                continue

            model_name = repo_id.split("/")[-1]
            artifact_id = f"{model_name.lower().replace('.', '')}_{args.tags[-1].lower()}_gguf".replace("-", "_")

            if artifact_id in artifacts_data["artifacts"]:
                print(f"  {artifact_id} already exists in artifacts.json")
                continue

            print(f"  Adding {artifact_id} ({size} bytes, publisher={publisher}, pipeline_tag={pipeline_tag})")

            sha256 = ""
            lfs = getattr(file, "lfs", None)
            if lfs:
                if isinstance(lfs, dict):
                    sha256 = lfs.get("sha256", "")
                else:
                    sha256 = getattr(lfs, "sha256", "")

            artifacts_data["artifacts"][artifact_id] = {
                "kind": "model",
                "framework": "llama.cpp-et",
                "variant": f"{model_name}-{args.tags[-1].upper()}",
                "filename": filename,
                "env": f"{artifact_id.upper()}_PATH",
                "source": {
                    "type": "huggingface",
                    "repo": repo_id,
                    "revision": "main",
                    "filename": filename,
                    "url": f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
                },
                "sha256": sha256,
                "size": size,
                "local_cache": f"local-artifacts/models/{filename}",
                "board_path": f"/data/models/{filename}"
            }

            bench_json = {
                "runner": "llama_server",
                "board": True,
                "benchmark_default": False,
                "framework": {
                    "name": "llama.cpp-et",
                    "runner": "llama_server",
                    "source_artifact": "llama_cpp_source"
                },
                "artifacts_file": "../artifacts.json",
                "canonical_variant": f"{model_name}-{args.tags[-1].upper()}",
                "score": {
                    "metric": "tokens_per_second",
                    "label": "Decode tokens/s",
                    "higher_is_better": True
                },
                "llama_server": {
                    "source_artifact": "llama_cpp_source",
                    "model_artifact": artifact_id,
                    "server_artifact": "llama_server",
                    "workdir_artifact": "llama_cpp_build",
                    "host": "127.0.0.1",
                    "port": 18081,
                    "device": "ET",
                    "gpu_layers": 1,
                    "ctx_size": 2048,
                    "batch_size": 256,
                    "ubatch_size": 128,
                    "parallel": 1,
                    "cache_ram_mib": 0,
                    "ready_timeout_s": 180,
                    "request_timeout_s": 300,
                    "flash_attn": False,
                    "api": "completion",
                    "prompt": "Repeat this token sequence without commentary: OK OK OK OK OK OK OK OK OK OK",
                    "max_tokens": 96,
                    "temperature": 0,
                    "ignore_eos": True,
                    "min_completion_tokens": 32,
                    "perplexity": {
                        "enabled": True,
                        "perplexity_artifact": "llama_perplexity",
                        "corpus_artifact": "wikitext2_raw_test",
                        "ctx_size": 128,
                        "batch_size": 128,
                        "ubatch_size": 128,
                        "timeout_s": 300,
                        "min_ppl": 1.0,
                        "max_ppl": 1000.0,
                        "chunks": 4
                    }
                }
            }

            bench_path = os.path.join(args.benchmarks, f"{model_name.lower().replace('.', '').replace('-', '_')}.json")
            with open(bench_path, "w") as bf:
                json.dump(bench_json, bf, indent=2)

            existing_identities.add(identity)
            count += 1
            break

    with open(args.artifacts, "w") as f:
        json.dump(artifacts_data, f, indent=2)

    print(f"Determinex Conveyor Auto-Porter added {count} models.")
    print(f"Skipped: {skipped}")
    print("NOTE: benchmark_default is written as False by design -- a human "
          "reviews and promotes each new candidate to the default board suite "
          "explicitly (see corpus 'benchmark_default_flag': this is shared CI "
          "config, never auto-promoted).")


if __name__ == "__main__":
    main()
