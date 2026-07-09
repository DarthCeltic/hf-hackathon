# YOLO End-to-End Detector

This port runs the real YOLOv10n detector path on ET-SoC1:

- raw uint8 RGB input at `0x04A00000`
- on-chip resize/normalization/transposition to 288x512 CHW FP32
- YOLOv10n backbone, neck, and detection heads
- on-chip DFL decode, class sigmoid, thresholding, and class-aware NMS
- compact detections at `0x01D00000`

The CI benchmark uses five committed raw RGB samples and gates every run on
expected category detections. The suite covers car/person, cat, giraffe,
elephant, and baseball-scene objects.
