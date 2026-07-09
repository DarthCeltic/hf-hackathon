# YOLO End-to-End Validation

The benchmark loads:

- `yolo/weights_region.bin` at `0x02000000`
- one of five raw RGB images at `0x04A00000`

The committed image cases are:

- `web_car`
- `coco_cat_524280`
- `coco_giraffes_296969`
- `coco_elephants_445248`
- `coco_baseball_043816`

The kernel writes detections at `0x01D00000`:

```c
uint32_t count;
struct {
    uint32_t class_id;
    float score;
    float x1;
    float y1;
    float x2;
    float y2;
} detections[count];
```

CI parses that list for every image case and requires the configured expected
categories, scores, and boxes from `.github/ci/benchmark_config.json`. The suite
currently checks:

- `car` (`class_id=2`) with score >= 0.55 and IoU >= 0.70 versus the expected box
- `person` (`class_id=0`) with score >= 0.35 and IoU >= 0.60 versus the expected box
- `cat` (`class_id=15`)
- `giraffe` (`class_id=23`)
- `elephant` (`class_id=20`)
- `baseball glove` (`class_id=35`)
- `sports ball` (`class_id=32`)

This catches the failure mode where weights or activations collapse to zero:
the detector emits no valid categories on one or more images, so the leaderboard
score fails even if the kernel exits successfully.
