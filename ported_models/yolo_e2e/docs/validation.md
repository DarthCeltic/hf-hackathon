# YOLO End-to-End Validation

The benchmark loads:

- `yolo-e2e/weights_region.bin` at `0x02000000`
- `yolo-e2e/web_car_raw_480x640x3_uint8_rgb.bin` at `0x04A00000`

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

CI parses that list and requires:

- `car` (`class_id=2`) with score >= 0.55 and IoU >= 0.70 versus the expected box
- `person` (`class_id=0`) with score >= 0.35 and IoU >= 0.60 versus the expected box

This catches the failure mode where weights or activations collapse to zero:
the detector emits no valid categories, so the leaderboard score fails even if
the kernel exits successfully.
