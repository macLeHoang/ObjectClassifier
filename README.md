# ObjectClassifier

A second-stage verifier for an object detector. Each detection is cropped, embedded with a
frozen ImageNet backbone (ResNet18 by default), and scored by a **binary SVM for its class**:
is this a correct detection or a false positive?

```
detector → boxes + class → crop_box → ResNet18 features → per-class SVM → (is_true, score)
```

The detector itself is not part of this folder.

## Files

| File | Purpose |
|---|---|
| `feature_extractor.py` | Backbone registry, `crop_box` (crop convention), letterbox preprocessing, batched embedding |
| `train_svm.py` | Crops folder → cached features → one calibrated `LinearSVC` per class → `verifier.joblib` |
| `infer.py` | `Verifier` class for use after the detector, plus a CLI to score a folder of crops |

```bash
pip install -r requirements.txt
```

## 1. Training data

```
data/
  person/
    true/   vid03__f0120_d2.jpg ...   # correct "person" detections
    false/  vid07__f0044_d0.jpg ...   # boxes the detector called "person" but are wrong
  car/
    true/ ...
    false/ ...
```

- **Make crops from the detector's own outputs**, not from ground-truth boxes. Run the detector
  on labelled images: a box that matches a ground-truth object of the same class (e.g. IoU ≥ 0.5)
  goes in `true/`, anything else in `false/`. The `false/` crops are exactly the mistakes the SVM
  learns to reject.
- **Crop with `crop_box` and the same `pad`** you pass to `train_svm.py --pad` (default 0.15),
  so training crops look like inference crops:
  ```python
  import cv2
  from feature_extractor import crop_box
  cv2.imwrite(f"data/{cls}/{label}/{video}__f{frame:05d}_d{i}.jpg", crop_box(frame_bgr, box_xyxy, pad=0.15))
  ```
- **Name files `<group>__<anything>`.** Crops sharing a group (video, scene, capture session)
  are kept on the same side of the train/val split, so validation numbers aren't inflated by
  near-duplicate frames. Change the separator with `--group-sep`.
- Each class needs at least `--min-per-label` (default 10) true and false crops, and at least
  5 groups (with the default `--val-frac 0.2`).

## 2. Train

```bash
python train_svm.py --data data/ --out models/verifier.joblib
```

For each class:
1. Embed the crops. Features are cached in `cache/<backbone>_<size>/<class>.npz`, and only new or
   changed files are embedded again on later runs.
2. Hold out a grouped validation split.
3. Grid-search `C` with grouped CV, scored by average precision.
4. Pick the **highest threshold that still keeps `--target-recall` (default 0.98) of the true
   crops** on validation.
5. Refit on all of the class's data.

Outputs:
- `models/verifier.joblib`: SVM pipelines, thresholds, and the feature settings (backbone,
  input size, preprocessing, pad).
- `models/verifier_report.json`: per-class validation AP, precision/recall at the threshold,
  the share of false positives rejected, and the confusion matrix.

Useful flags: `--backbone resnet50`, `--target-recall 0.95` (rejects more false positives but loses
more true ones), `--device cuda:1`, `--workers 8`.

## 3. Inference

```python
import cv2
from ultralytics import YOLO
from infer import Verifier

detector = YOLO("yolo11n.pt")
verifier = Verifier.load("models/verifier.joblib")   # unknown="accept" | "reject"

frame = cv2.imread("image.jpg")                       # BGR
r = detector(frame)[0]
names = [r.names[int(c)] for c in r.boxes.cls]        # must match the data/<class> folder names
results = verifier.verify(frame, r.boxes.xyxy.cpu().numpy(), names, bgr=True)
keep = [i for i, res in enumerate(results) if res.is_true]
```

- All crops in a frame are embedded in one batched forward pass. Each class's SVM then scores
  its own rows.
- `score` is a calibrated probability that the detection is correct. `is_true` is
  `score >= threshold` for that class.
- Classes without a trained SVM return `score=None`. `is_true` is then `True` with
  `unknown="accept"` and `False` with `unknown="reject"`.
- Loading fails with a clear error if the saved feature settings don't match the current code.
  For example, an SVM trained on ResNet18 features can't run with a different backbone.

Spot-check a folder of crops from the command line:

```bash
python infer.py --model models/verifier.joblib --crops data/person/false --cls person --out scores.csv
```

## Adding a backbone

Add a factory to `BACKBONES` in `feature_extractor.py` that returns
`(model_outputting_pooled_features, feature_dim)`, then train with `--backbone <name>`.
