# YOLO Reference Model

This folder contains the YOLO kernel source, variant manifests, parse scripts,
and optimization notes used for ET-SoC1 sweeps.

CI uses the committed deterministic assets in `assets/yolo-bench/`:
four 80x80x16 float32 inputs, sparse float32 weights, and the expected
4x80x80x16 uint8 output tensor. Regenerate them with
`scripts/gen_yolo_image_set.py`.

The Hugging Face model package should be treated as the external reference
source. Keep ONNX/input blobs and generated ELFs out of git. The current base is
`onnx-community/yolov10n` pinned in
[`docs/HF_REFERENCES.md`](../../docs/HF_REFERENCES.md).

Start with `docs/optimizations.md`.
