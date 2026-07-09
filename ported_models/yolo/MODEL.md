# YOLOv10n End-to-End

| Field | Value |
| --- | --- |
| Model | YOLOv10n detector |
| Input | `uint8[480][640][3]` RGB, HWC |
| Preprocess | Bilinear resize to 288x512, scale by 1/255, HWC to CHW |
| Output | `{count, class_id, score, x1, y1, x2, y2}` detections |
| CI metric | End-to-end kernel wait seconds |
| CI accuracy | Required `car` and `person` detections on `web_car` |

This is the canonical `yolo` leaderboard benchmark. It proves that a real image
flows through the detector and produces the expected categories, rather than
only checking that an intermediate tensor has a fixed value.
