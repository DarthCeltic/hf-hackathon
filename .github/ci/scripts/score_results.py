#!/usr/bin/env python3
"""Score sys-emu benchmark output for one model."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import struct
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from benchmark_config_helpers import load_config as load_benchmark_config

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"

DNCNN_MAGIC = 0xD3C11003
YOLO_MAGIC = 0x10500001
SUMMARY = struct.Struct("<16I")
YOLO_DETECTION = struct.Struct("<I5f")


def int_cfg(value: int | str) -> int:
    if isinstance(value, int):
        return value
    return int(str(value), 0)


def load_config() -> dict:
    return load_benchmark_config(os.environ.get("BENCHMARK_CONFIG", CONFIG_PATH))


def wait_seconds(log_path: Path) -> float | None:
    text = log_path.read_text(errors="ignore")
    match = re.search(r"Kernel wait seconds:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def dump_summary(path: Path, model: str = "") -> dict:
    data = path.read_bytes()[: 0x1000 + SUMMARY.size]
    fields = SUMMARY.unpack_from(data, 0x1000)
    if model == "whisper":
        return {
            "magic": fields[0],
            "active_harts": fields[1],
            "passes": fields[2],
            "done_count": fields[7],
            "output_sum": fields[8],
            "slot_sum": fields[9],
            "ops": fields[10] | (fields[11] << 32),
        }
    return {
        "magic": fields[0],
        "active_harts": fields[1],
        "passes": fields[2],
        "done_count": fields[8],
        "output_sum": fields[9],
        "slot_sum": fields[10],
        "ops": fields[11] | (fields[12] << 32),
    }


def read_dump_bytes(path: Path, offset: int, count: int) -> bytes:
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read(count)
    if len(data) != count:
        raise ValueError(f"short dump read at 0x{offset:x}: got {len(data)} bytes, expected {count}")
    return data


def read_dump_text(path: Path, offset: int, count: int, encoding: str = "utf-8") -> str:
    data = read_dump_bytes(path, offset, count)
    data = data.split(b"\0", 1)[0]
    return data.decode(encoding, errors="replace")


def normalize_transcript(text: str, mode: str | bool | None = "whitespace") -> str:
    if mode in (False, "none", "raw"):
        return text
    normalized = unicodedata.normalize("NFKC", text).replace("\x00", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if mode in (True, None, "whitespace"):
        return normalized
    if mode in ("lower", "lowercase", "lower_whitespace"):
        return normalized.lower()
    raise ValueError(f"unknown transcript normalization mode {mode!r}")


def cell_text(text: str, limit: int = 80) -> str:
    flat = " ".join(text.split())
    return (flat[: limit - 1] + "...") if len(flat) > limit else flat


def with_metrics(metrics: dict, **updates) -> dict:
    merged = dict(metrics)
    merged.update(updates)
    return merged


def load_uint8_npy(path: Path) -> tuple[tuple[int, ...], bytes]:
    data = path.read_bytes()
    if not data.startswith(b"\x93NUMPY"):
        raise ValueError("reference is not a .npy file")
    major = data[6]
    if major == 1:
        header_len = struct.unpack_from("<H", data, 8)[0]
        header_start = 10
    elif major in (2, 3):
        header_len = struct.unpack_from("<I", data, 8)[0]
        header_start = 12
    else:
        raise ValueError(f"unsupported .npy version {major}.{data[7]}")

    header = ast.literal_eval(data[header_start : header_start + header_len].decode("latin1").strip())
    if header.get("fortran_order"):
        raise ValueError("fortran-order .npy references are not supported")
    if header.get("descr") not in ("|u1", "u1"):
        raise ValueError(f"expected uint8 .npy reference, got {header.get('descr')}")
    shape = tuple(int(v) for v in header["shape"])
    count = math.prod(shape)
    payload = data[header_start + header_len :]
    if len(payload) != count:
        raise ValueError(f"reference payload has {len(payload)} bytes, expected {count}")
    return shape, payload


def artifact_root() -> Path:
    return Path(
        os.environ.get("BENCHMARK_ARTIFACT_ROOT")
        or os.environ.get("AMP_ROOT")
        or REPO_ROOT / "local-artifacts" / "model-port-benchmarks"
    )


def resolve_reference(paths: list[str]) -> Path | None:
    roots = [artifact_root(), REPO_ROOT]
    for rel in paths:
        candidate = Path(rel)
        if candidate.is_absolute() and candidate.is_file():
            return candidate
        for root in roots:
            path = root / rel
            if path.is_file():
                return path
    return None


def byte_diff_metrics(actual: bytes, expected: bytes) -> tuple[int, float]:
    if len(actual) != len(expected):
        raise ValueError(f"length mismatch: actual {len(actual)} bytes, expected {len(expected)} bytes")
    max_abs = 0
    total = 0
    for a, b in zip(actual, expected):
        diff = abs(a - b)
        total += diff
        if diff > max_abs:
            max_abs = diff
    mean_abs = total / len(actual) if actual else 0.0
    return max_abs, mean_abs


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def max_box_abs_diff(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def read_yolo_detections(path: Path, offset: int, max_detections: int) -> list[dict]:
    count = struct.unpack("<I", read_dump_bytes(path, offset, 4))[0]
    if count > max_detections:
        raise ValueError(f"detection count {count} exceeds configured max {max_detections}")
    payload = read_dump_bytes(path, offset + 4, count * YOLO_DETECTION.size)
    detections = []
    for idx in range(count):
        class_id, score, x1, y1, x2, y2 = YOLO_DETECTION.unpack_from(payload, idx * YOLO_DETECTION.size)
        detections.append(
            {
                "class_id": int(class_id),
                "score": float(score),
                "box": (float(x1), float(y1), float(x2), float(y2)),
            }
        )
    return detections


def validate_accuracy(
    model: str,
    dump_path: Path | None,
    mcfg: dict,
    summary: dict | None,
    accuracy_cfg: dict | None = None,
) -> tuple[bool, str, dict]:
    acfg = accuracy_cfg if accuracy_cfg is not None else mcfg.get("accuracy")
    if not acfg:
        return True, "no accuracy gate configured", {
            "valid_accuracy": True,
            "accuracy_kind": None,
            "accuracy_max_abs": None,
            "accuracy_mean_abs": None,
            "accuracy_reference": None,
        }

    kind = acfg.get("kind")
    metrics = {
        "valid_accuracy": False,
        "accuracy_kind": kind,
        "accuracy_max_abs": None,
        "accuracy_mean_abs": None,
        "accuracy_reference": None,
    }

    try:
        if kind == "checksum":
            if summary is None:
                return False, "accuracy check failed: missing dump summary", metrics
            expected = int_cfg(acfg["expected_output_sum"])
            actual = int(summary.get("output_sum", -1))
            ok = actual == expected
            note = f"accuracy checksum {'valid' if ok else 'failed'} output_sum={actual} expected={expected}"
            return ok, note, with_metrics(metrics, valid_accuracy=ok)

        if kind == "sha256":
            if dump_path is None or not dump_path.is_file():
                return False, "accuracy check failed: missing dump.bin", metrics
            offset = int_cfg(acfg["offset"])
            count = int_cfg(acfg["count"])
            expected = str(acfg["expected_sha256"]).lower()
            actual = hashlib.sha256(read_dump_bytes(dump_path, offset, count)).hexdigest()
            ok = actual == expected
            note = (
                f"accuracy sha256 {'valid' if ok else 'failed'} "
                f"actual={actual[:12]} expected={expected[:12]}"
            )
            return ok, note, with_metrics(metrics, valid_accuracy=ok)

        if kind == "constant_u8":
            if dump_path is None or not dump_path.is_file():
                return False, "accuracy check failed: missing dump.bin", metrics
            offset = int_cfg(acfg["offset"])
            count = int_cfg(acfg["count"])
            expected_value = int_cfg(acfg["expected_value"])
            max_allowed = int_cfg(acfg.get("max_abs", 0))
            actual = read_dump_bytes(dump_path, offset, count)
            expected = bytes([expected_value]) * count
            max_abs, mean_abs = byte_diff_metrics(actual, expected)
            ok = max_abs <= max_allowed
            note = (
                f"accuracy {'valid' if ok else 'failed'} constant_u8 "
                f"max_abs={max_abs} mean_abs={mean_abs:.6f} gate={max_allowed}"
            )
            return ok, note, with_metrics(
                metrics,
                valid_accuracy=ok,
                accuracy_max_abs=max_abs,
                accuracy_mean_abs=mean_abs,
            )

        if kind == "uint8_npy":
            if dump_path is None or not dump_path.is_file():
                return False, "accuracy check failed: missing dump.bin", metrics
            ref_paths = acfg.get("reference_paths") or [acfg.get("reference_path")]
            ref_paths = [str(path) for path in ref_paths if path]
            ref_path = resolve_reference(ref_paths)
            if ref_path is None:
                return False, "accuracy check failed: missing reference " + ", ".join(ref_paths), metrics
            shape, expected = load_uint8_npy(ref_path)
            cfg_shape = tuple(int(v) for v in acfg.get("shape", shape))
            if shape != cfg_shape:
                return False, f"accuracy check failed: reference shape {shape} != configured {cfg_shape}", metrics
            actual = read_dump_bytes(dump_path, int_cfg(acfg["offset"]), len(expected))
            max_abs, mean_abs = byte_diff_metrics(actual, expected)
            max_allowed = int_cfg(acfg.get("max_abs", 0))
            ok = max_abs <= max_allowed
            rel = str(ref_path)
            try:
                rel = str(ref_path.relative_to(REPO_ROOT))
            except ValueError:
                pass
            note = (
                f"accuracy {'valid' if ok else 'failed'} uint8_npy "
                f"max_abs={max_abs} mean_abs={mean_abs:.6f} gate={max_allowed}"
            )
            return ok, note, with_metrics(
                metrics,
                valid_accuracy=ok,
                accuracy_max_abs=max_abs,
                accuracy_mean_abs=mean_abs,
                accuracy_reference=rel,
            )

        if kind == "yolo_detections":
            if dump_path is None or not dump_path.is_file():
                return False, "accuracy check failed: missing dump.bin", metrics
            offset = int_cfg(acfg["offset"])
            max_detections = int_cfg(acfg.get("max_detections", 64))
            detections = read_yolo_detections(dump_path, offset, max_detections)
            expected = acfg.get("expected", [])
            if not isinstance(expected, list) or not expected:
                return False, "accuracy check failed: yolo_detections has no expected entries", metrics

            used: set[int] = set()
            matches: list[str] = []
            failures: list[str] = []
            box_abs_values: list[float] = []

            for exp in expected:
                class_id = int_cfg(exp["class_id"])
                label = str(exp.get("label") or f"class_{class_id}")
                min_score = float(exp.get("min_score", acfg.get("min_score", 0.0)))
                exp_box = exp.get("box")
                exp_box_tuple = tuple(float(v) for v in exp_box) if exp_box is not None else None

                candidates = [
                    (idx, det)
                    for idx, det in enumerate(detections)
                    if idx not in used
                    and int(det["class_id"]) == class_id
                    and float(det["score"]) >= min_score
                ]
                if not candidates:
                    failures.append(f"{label}: missing class_id={class_id} score>={min_score:.2f}")
                    continue

                if exp_box_tuple is None:
                    best_idx, best_det = max(candidates, key=lambda item: float(item[1]["score"]))
                    used.add(best_idx)
                    matches.append(f"{label} score={float(best_det['score']):.3f}")
                    continue

                scored = []
                for idx, det in candidates:
                    det_box = det["box"]
                    iou = box_iou(det_box, exp_box_tuple)
                    abs_diff = max_box_abs_diff(det_box, exp_box_tuple)
                    scored.append((iou, -abs_diff, idx, det, abs_diff))
                iou, neg_abs_diff, best_idx, best_det, abs_diff = max(scored, key=lambda item: (item[0], item[1]))
                min_iou = float(exp.get("min_iou", acfg.get("min_iou", 0.5)))
                max_abs_allowed = float(exp.get("max_box_abs", acfg.get("max_box_abs", "inf")))
                if iou < min_iou:
                    failures.append(f"{label}: iou={iou:.3f} below {min_iou:.3f}")
                    continue
                if abs_diff > max_abs_allowed:
                    failures.append(f"{label}: box_abs={abs_diff:.2f} above {max_abs_allowed:.2f}")
                    continue
                used.add(best_idx)
                box_abs_values.append(abs_diff)
                matches.append(
                    f"{label} score={float(best_det['score']):.3f} iou={iou:.3f} box_abs={abs_diff:.1f}"
                )

            ok = not failures
            found = ", ".join(matches) if matches else "none"
            if failures:
                found += "; " if found else ""
                found += "failures: " + "; ".join(failures)
            note = (
                f"accuracy yolo_detections {'valid' if ok else 'failed'} "
                f"count={len(detections)} found={found}"
            )
            max_abs = max(box_abs_values) if box_abs_values else None
            mean_abs = sum(box_abs_values) / len(box_abs_values) if box_abs_values else None
            return ok, note, with_metrics(
                metrics,
                valid_accuracy=ok,
                accuracy_max_abs=max_abs,
                accuracy_mean_abs=mean_abs,
                accuracy_reference=acfg.get("reference"),
            )

        if kind in ("transcript", "transcript_exact"):
            if dump_path is None or not dump_path.is_file():
                return False, "accuracy check failed: missing dump.bin", metrics
            offset = int_cfg(acfg["offset"])
            count = int_cfg(acfg.get("count", acfg.get("max_bytes", 4096)))
            encoding = str(acfg.get("encoding", "utf-8"))
            actual = read_dump_text(dump_path, offset, count, encoding=encoding)

            expected_ref = None
            if "expected_text" in acfg:
                expected = str(acfg["expected_text"])
                expected_ref = "inline expected_text"
            else:
                ref_paths = acfg.get("expected_text_paths") or acfg.get("reference_paths") or [
                    acfg.get("expected_text_path") or acfg.get("reference_path")
                ]
                ref_paths = [str(path) for path in ref_paths if path]
                ref_path = resolve_reference(ref_paths)
                if ref_path is None:
                    return False, "accuracy check failed: missing transcript reference " + ", ".join(ref_paths), metrics
                expected = ref_path.read_text(encoding=encoding)
                expected_ref = str(ref_path)
                try:
                    expected_ref = str(ref_path.relative_to(REPO_ROOT))
                except ValueError:
                    pass

            mode = acfg.get("normalize", "whitespace")
            actual_norm = normalize_transcript(actual, mode)
            expected_norm = normalize_transcript(expected, mode)
            ok = actual_norm == expected_norm
            actual_preview = cell_text(actual_norm)
            expected_preview = cell_text(expected_norm)
            note = (
                f"accuracy transcript_exact {'valid' if ok else 'failed'} "
                f"actual={actual_preview!r} expected={expected_preview!r}"
            )
            return ok, note, with_metrics(
                metrics,
                valid_accuracy=ok,
                accuracy_reference=expected_ref,
                accuracy_text=actual_norm,
                accuracy_expected_text=expected_norm,
            )

        return False, f"accuracy check failed: unknown kind {kind}", metrics
    except Exception as exc:
        return False, f"accuracy check failed: {exc}", metrics


def validate_dump(model: str, dump_path: Path | None, magic_cfg: str | None) -> tuple[bool, str]:
    if magic_cfg is None:
        return True, "no dump magic configured"
    if dump_path is None or not dump_path.is_file():
        return False, "missing dump.bin"
    magic_expected = int(magic_cfg, 0)
    s = dump_summary(dump_path, model)
    ok = (
        s["magic"] == magic_expected
        and s["done_count"] == s["active_harts"]
        and s["output_sum"] == s["slot_sum"]
    )
    if not ok:
        return False, (
            f"dump check failed magic=0x{s['magic']:x} "
            f"done={s['done_count']} harts={s['active_harts']} "
            f"out={s['output_sum']} slot={s['slot_sum']}"
        )
    return True, "dump valid"


def benchmark_image_count(mcfg: dict) -> int | None:
    for section in ("accuracy", "validation"):
        cfg = mcfg.get(section, {})
        if not isinstance(cfg, dict) or "image_count" not in cfg:
            continue
        try:
            value = int(cfg["image_count"])
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def find_job_dir(results_dir: Path, model: str, variant: str) -> Path | None:
    jobs = results_dir / "jobs"
    if not jobs.is_dir():
        return None
    for job in sorted(jobs.iterdir()):
        if variant in job.name and model in job.name:
            return job
    return None


def row_paths(results_dir: Path, row: dict, job_dir: Path | None = None) -> tuple[Path | None, Path | None]:
    log_path = (results_dir / row["log"]) if row.get("log") else None
    dump_path = (results_dir / row["dump"]) if row.get("dump") else None
    if job_dir and (job_dir / "run.log").is_file():
        log_path = job_dir / "run.log"
    if job_dir and (job_dir / "dump.bin").is_file():
        dump_path = job_dir / "dump.bin"
    return log_path, dump_path


def evaluate_row(
    model: str,
    mcfg: dict,
    row: dict,
    results_dir: Path,
    magic_cfg: str | None,
    accuracy_cfg: dict | None = None,
    job_dir: Path | None = None,
) -> dict:
    status = row.get("status", "")
    log_path, dump_path = row_paths(results_dir, row, job_dir)
    if status == "fail" and log_path and log_path.is_file():
        log_text = log_path.read_text(errors="ignore")
        if "kernel execution timed out" in log_text or "timed out" in log_text.lower():
            status = "timeout"

    kernel_wait = row.get("kernel_wait_s") or ""
    if not kernel_wait and log_path and log_path.is_file():
        w = wait_seconds(log_path)
        kernel_wait = f"{w:.6f}" if w is not None else ""

    valid_dump, valid_note = validate_dump(model, dump_path, magic_cfg)
    parsed_summary = dump_summary(dump_path, model) if dump_path and dump_path.is_file() else None
    valid_accuracy, accuracy_note, accuracy_metrics = validate_accuracy(
        model,
        dump_path,
        mcfg,
        parsed_summary,
        accuracy_cfg=accuracy_cfg,
    )
    combined_note = valid_note
    if accuracy_note:
        combined_note = f"{valid_note}; {accuracy_note}"
    passed = status == "pass" and bool(kernel_wait) and valid_dump and valid_accuracy
    return {
        "case": row.get("case") or "",
        "status": "pass" if passed else (status if status and status != "pass" else "fail"),
        "passed": passed,
        "kernel_wait_s": float(kernel_wait) if kernel_wait else None,
        "valid_dump": valid_dump,
        "valid_accuracy": valid_accuracy,
        "valid_note": combined_note,
        "accuracy_metrics": accuracy_metrics,
        "emu_cycle_last": row.get("emu_cycle_last") or None,
        "elapsed_s": float(row["elapsed_s"]) if row.get("elapsed_s") else None,
        "note": row.get("note") or "",
    }


def score_benchmark_cases(
    model: str,
    mcfg: dict,
    variant: str,
    magic_cfg: str | None,
    rows: list[dict],
    results_dir: Path,
    sha: str,
    ref: str,
    actor: str,
    run_url: str,
) -> dict | None:
    cases = [case for case in mcfg.get("benchmark_cases", []) if isinstance(case, dict) and case.get("name")]
    if not cases:
        return None

    rows_by_case = {
        row.get("case"): row
        for row in rows
        if row.get("model") == model and row.get("variant") == variant and row.get("case")
    }
    case_results = []
    for case in cases:
        name = str(case["name"])
        row = rows_by_case.get(name)
        if row is None:
            case_results.append(
                {
                    "case": name,
                    "status": "missing",
                    "passed": False,
                    "kernel_wait_s": None,
                    "valid_dump": False,
                    "valid_accuracy": False,
                    "valid_note": "missing benchmark case row",
                    "accuracy_metrics": {
                        "accuracy_kind": case.get("accuracy", {}).get("kind"),
                        "accuracy_max_abs": None,
                        "accuracy_mean_abs": None,
                        "accuracy_reference": case.get("accuracy", {}).get("reference"),
                    },
                    "emu_cycle_last": None,
                    "elapsed_s": None,
                    "note": "missing benchmark case row",
                }
            )
            continue
        case_results.append(
            evaluate_row(
                model,
                mcfg,
                row,
                results_dir,
                magic_cfg,
                accuracy_cfg=case.get("accuracy") if isinstance(case.get("accuracy"), dict) else None,
            )
        )

    passed = all(result["passed"] for result in case_results)
    waits = [result["kernel_wait_s"] for result in case_results if isinstance(result["kernel_wait_s"], float)]
    kernel_wait_value = sum(waits) / len(waits) if waits else None
    elapsed_values = [result["elapsed_s"] for result in case_results if isinstance(result["elapsed_s"], float)]
    elapsed_s = sum(elapsed_values) if elapsed_values else None
    failed_notes = [
        f"{result['case']}: {result['valid_note'] or result['status']}"
        for result in case_results
        if not result["passed"]
    ]
    valid_note = (
        f"{sum(1 for result in case_results if result['passed'])}/{len(case_results)} "
        "YOLO image cases valid"
    )
    if failed_notes:
        valid_note += "; " + "; ".join(failed_notes[:3])

    accuracy_metrics = [result["accuracy_metrics"] for result in case_results]
    max_abs_values = [
        metric.get("accuracy_max_abs")
        for metric in accuracy_metrics
        if isinstance(metric.get("accuracy_max_abs"), (int, float))
    ]
    mean_abs_values = [
        metric.get("accuracy_mean_abs")
        for metric in accuracy_metrics
        if isinstance(metric.get("accuracy_mean_abs"), (int, float))
    ]
    first_kind = next((metric.get("accuracy_kind") for metric in accuracy_metrics if metric.get("accuracy_kind")), None)
    return {
        "model": model,
        "variant": variant,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "kernel_wait_s": kernel_wait_value,
        "kernel_wait_per_image_s": kernel_wait_value,
        "tokens_per_second": None,
        "prompt_tokens_per_second": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "perplexity": None,
        "perplexity_error": None,
        "perplexity_tokens": None,
        "perplexity_prompt_tokens_per_second": None,
        "valid_dump": all(result["valid_dump"] for result in case_results),
        "valid_note": valid_note,
        "valid_accuracy": all(result["valid_accuracy"] for result in case_results),
        "accuracy_kind": first_kind,
        "accuracy_max_abs": max(max_abs_values) if max_abs_values else None,
        "accuracy_mean_abs": sum(mean_abs_values) / len(mean_abs_values) if mean_abs_values else None,
        "accuracy_reference": f"{len(case_results)}-image YOLO detection suite",
        "case_results": [
            {
                key: value
                for key, value in result.items()
                if key != "accuracy_metrics"
            }
            for result in case_results
        ],
        "emu_cycle_last": None,
        "elapsed_s": elapsed_s,
        "note": "",
        "sha": sha,
        "ref": ref,
        "team": actor,
        "run_url": run_url,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def score_from_results(
    model: str,
    results_dir: Path,
    sha: str,
    ref: str,
    actor: str,
    run_url: str,
) -> dict:
    cfg = load_config()
    mcfg = cfg["models"][model]
    variant = mcfg["canonical_variant"]
    magic_cfg = mcfg.get("dump_magic")

    results_tsv = results_dir / "results.tsv"
    if not results_tsv.is_file():
        return fail_payload(model, variant, sha, ref, actor, run_url, "missing results.tsv")

    rows = []
    with results_tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)

    case_score = score_benchmark_cases(
        model,
        mcfg,
        variant,
        magic_cfg,
        rows,
        results_dir,
        sha,
        ref,
        actor,
        run_url,
    )
    if case_score is not None:
        return case_score

    row = None
    for r in rows:
        if r.get("model") == model and r.get("variant") == variant:
            row = r
            break

    if row is None:
        return fail_payload(model, variant, sha, ref, actor, run_url, "canonical variant not in results.tsv")

    job_dir = find_job_dir(results_dir, model, variant)
    evaluated = evaluate_row(model, mcfg, row, results_dir, magic_cfg, job_dir=job_dir)
    kernel_wait_value = evaluated["kernel_wait_s"]
    image_count = benchmark_image_count(mcfg)
    kernel_wait_per_image = (
        kernel_wait_value / image_count
        if kernel_wait_value is not None and image_count
        else None
    )

    return {
        "model": model,
        "variant": variant,
        "status": evaluated["status"],
        "passed": evaluated["passed"],
        "kernel_wait_s": kernel_wait_value,
        "kernel_wait_per_image_s": kernel_wait_per_image,
        "tokens_per_second": None,
        "prompt_tokens_per_second": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "perplexity": None,
        "perplexity_error": None,
        "perplexity_tokens": None,
        "perplexity_prompt_tokens_per_second": None,
        "valid_dump": evaluated["valid_dump"],
        "valid_note": evaluated["valid_note"],
        **evaluated["accuracy_metrics"],
        "emu_cycle_last": evaluated["emu_cycle_last"],
        "elapsed_s": evaluated["elapsed_s"],
        "note": evaluated["note"],
        "sha": sha,
        "ref": ref,
        "team": actor,
        "run_url": run_url,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def fail_payload(
    model: str,
    variant: str,
    sha: str,
    ref: str,
    actor: str,
    run_url: str,
    note: str,
    status: str = "fail",
) -> dict:
    return {
        "model": model,
        "variant": variant,
        "status": status,
        "passed": False,
        "kernel_wait_s": None,
        "kernel_wait_per_image_s": None,
        "tokens_per_second": None,
        "prompt_tokens_per_second": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "perplexity": None,
        "perplexity_error": None,
        "perplexity_tokens": None,
        "perplexity_prompt_tokens_per_second": None,
        "valid_dump": False,
        "valid_accuracy": False,
        "accuracy_kind": None,
        "accuracy_max_abs": None,
        "accuracy_mean_abs": None,
        "accuracy_reference": None,
        "valid_note": note,
        "emu_cycle_last": None,
        "elapsed_s": None,
        "note": note,
        "sha": sha,
        "ref": ref,
        "team": actor,
        "run_url": run_url,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--status")
    parser.add_argument("--note", default="")
    parser.add_argument("--sha", default="local")
    parser.add_argument("--ref", default="local")
    parser.add_argument("--actor", default="local")
    parser.add_argument("--run-url", default="")
    args = parser.parse_args()

    if args.status:
        cfg = load_config()
        variant = cfg["models"][args.model]["canonical_variant"]
        payload = fail_payload(
            args.model,
            variant,
            args.sha,
            args.ref,
            args.actor,
            args.run_url,
            args.note,
            status=args.status,
        )
    else:
        if not args.results_dir:
            print("error: --results-dir required unless --status is set", file=sys.stderr)
            return 2
        payload = score_from_results(
            args.model,
            Path(args.results_dir),
            args.sha,
            args.ref,
            args.actor,
            args.run_url,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
