# YOLO Model Card

- Reference family: YOLOv10n object detection.
- Hugging Face base: `onnx-community/yolov10n` at
  `57657320425ee34056408a57ad9d29c4d4815bd8`.
- Reference file: `onnx/model.onnx`.
- Main source: `src/yolo_vpu_argbuf.c`.
- Smoke manifest: `manifests/yolo_10_variants.txt`.
- Sweep manifest: `manifests/yolo_100_variants.txt`.
- CI accuracy assets: `assets/yolo-bench/yolo_image_set_input_f32.bin`,
  `assets/yolo-bench/yolo_image_set_weights_f32.bin`, and
  `assets/yolo-bench/yolo_image_set_output_u8.npy`.
- Key docs: `docs/optimizations.md`.

The CI row runs four deterministic 80x80x16 feature-map inputs with preloaded
weights and compares the full 4x80x80x16 uint8 output tensor against the
committed reference. The leaderboard metric is kernel wait per image so this
four-image validation row is not compared against the old one-image smoke time.
Real-image category validation is covered separately by `ported_models/yolo_e2e`.
Use the pinned Hugging Face model config/preprocessor metadata for model I/O and
keep large ONNX artifacts outside git.
