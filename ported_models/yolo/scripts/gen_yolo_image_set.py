#!/usr/bin/env python3
"""Generate deterministic YOLO image-set benchmark assets."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

IMG_W = 80
IMG_H = 80
CH = 16
HEAD_CH = 16
K = 3
BLOCKS = 4
IMAGE_COUNT = 4

ACT_FLOATS = IMG_W * IMG_H * CH
OUT_BYTES = IMG_W * IMG_H * HEAD_CH
CONV3_WEIGHTS = CH * K * K * CH
CONV1_WEIGHTS = CH * CH
BLOCK_WEIGHTS = CONV3_WEIGHTS + CONV1_WEIGHTS
HEAD_WEIGHTS = HEAD_CH * CH
WEIGHT_FLOATS = BLOCKS * BLOCK_WEIGHTS + HEAD_WEIGHTS


def input_code(image: int, y: int, x: int, c: int) -> int:
    return (image * 17 + y * 5 + x * 3 + c * 11) & 63


def clamp(v: int, limit: int) -> int:
    if v < 0:
        return 0
    if v >= limit:
        return limit - 1
    return v


def conv3_shift(block: int, channel: int) -> tuple[int, int]:
    pos = (channel + block) % (K * K)
    return pos // K - 1, pos % K - 1


def write_npy_u8(path: Path, shape: tuple[int, ...], payload: bytes) -> None:
    if len(payload) != prod(shape):
        raise ValueError(f"payload length {len(payload)} does not match shape {shape}")
    shape_text = "(" + ", ".join(str(v) for v in shape)
    shape_text += ",)" if len(shape) == 1 else ")"
    header = f"{{'descr': '|u1', 'fortran_order': False, 'shape': {shape_text}, }}"
    padding = 16 - ((10 + len(header) + 1) % 16)
    header = header + (" " * padding) + "\n"
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin1") + payload)


def prod(values: tuple[int, ...]) -> int:
    out = 1
    for value in values:
        out *= value
    return out


def make_inputs() -> bytes:
    out = bytearray()
    pack = struct.Struct("<f").pack
    for image in range(IMAGE_COUNT):
        for y in range(IMG_H):
            for x in range(IMG_W):
                for c in range(CH):
                    out.extend(pack(input_code(image, y, x, c) / 64.0))
    return bytes(out)


def set_f32(buf: bytearray, index: int, value: float) -> None:
    struct.pack_into("<f", buf, index * 4, value)


def make_weights() -> bytes:
    out = bytearray(WEIGHT_FLOATS * 4)
    for block in range(BLOCKS):
        block_base = block * BLOCK_WEIGHTS
        conv3_base = block_base
        conv1_base = block_base + CONV3_WEIGHTS
        for oc in range(CH):
            pos = (oc + block) % (K * K)
            set_f32(out, conv3_base + oc * K * K * CH + pos * CH + oc, 256.0)
            set_f32(out, conv1_base + oc * CH + oc, 128.0)

    head_base = BLOCKS * BLOCK_WEIGHTS
    for oc in range(HEAD_CH):
        set_f32(out, head_base + oc * CH + oc, 4096.0)
    return bytes(out)


def make_reference() -> bytes:
    out = bytearray(IMAGE_COUNT * OUT_BYTES)
    offset = 0
    for image in range(IMAGE_COUNT):
        for y in range(IMG_H):
            for x in range(IMG_W):
                for c in range(HEAD_CH):
                    src_y = y
                    src_x = x
                    for block in range(BLOCKS - 1, -1, -1):
                        dy, dx = conv3_shift(block, c)
                        src_y = clamp(src_y + dy, IMG_H)
                        src_x = clamp(src_x + dx, IMG_W)
                    out[offset] = 128 + input_code(image, src_y, src_x, c)
                    offset += 1
    return bytes(out)


def write_asset(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"{path}: {len(data)} bytes sha256={digest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parents[1] / "assets" / "yolo-bench",
        type=Path,
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    write_asset(out_dir / "yolo_image_set_input_f32.bin", make_inputs())
    write_asset(out_dir / "yolo_image_set_weights_f32.bin", make_weights())
    ref = make_reference()
    write_npy_u8(out_dir / "yolo_image_set_output_u8.npy", (IMAGE_COUNT, IMG_H, IMG_W, HEAD_CH), ref)
    digest = hashlib.sha256((out_dir / "yolo_image_set_output_u8.npy").read_bytes()).hexdigest()
    print(f"{out_dir / 'yolo_image_set_output_u8.npy'}: sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
