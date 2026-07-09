#!/usr/bin/env python3
import argparse
import json
import os
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Determinex LLM Auto-Porter for ET-SoC1")
    parser.add_argument("--tags", nargs='+', default=["gguf", "q8_0"], help="Tags to filter models")
    parser.add_argument("--max-size", type=int, default=3000000000, help="Max model size in bytes (3GB)")
    parser.add_argument("--limit", type=int, default=10, help="Number of models to process")
    parser.add_argument("--artifacts", type=str, default="ported_models/llama_cpp_et/artifacts.json", help="Path to artifacts.json")
    parser.add_argument("--benchmarks", type=str, default="ported_models/llama_cpp_et/benchmarks", help="Path to benchmarks dir")
    args = parser.parse_args()

    api = HfApi()
    models = api.list_models(filter="gguf", sort="downloads", limit=100)
    
    with open(args.artifacts, "r") as f:
        artifacts_data = json.load(f)
        
    count = 0
    for m in models:
        if count >= args.limit:
            break
        repo_id = m.id
        print(f"Checking {repo_id}...")
        
        try:
            info = api.model_info(repo_id, files_metadata=True)
        except Exception as e:
            print(f"Error fetching info for {repo_id}: {e}")
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
                continue
                
            model_name = repo_id.split("/")[-1]
            artifact_id = f"{model_name.lower().replace('.', '')}_{args.tags[-1].lower()}_gguf".replace("-", "_")
            
            if artifact_id in artifacts_data["artifacts"]:
                print(f"  {artifact_id} already exists in artifacts.json")
                continue
                
            print(f"  Adding {artifact_id} ({size} bytes)")
            
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
                
            count += 1
            break
            
    with open(args.artifacts, "w") as f:
        json.dump(artifacts_data, f, indent=2)

    print(f"Determinex Conveyor Auto-Porter added {count} models.")

if __name__ == "__main__":
    main()
