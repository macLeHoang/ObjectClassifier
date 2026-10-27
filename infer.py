"""Verify detector outputs with the per-class SVMs trained by train_svm.py.

Library use:
    v = Verifier.load("models/verifier.joblib")
    results = v.verify(frame, boxes_xyxy, class_names, bgr=True)   # [VerifyResult(is_true, score)]

CLI (spot-check a folder of crops):
    python infer.py --model models/verifier.joblib --crops data/person/false --cls person
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Sequence

import joblib
import numpy as np

from feature_extractor import IMG_EXTS, FeatureExtractor, crop_box, load_rgb

# Feature settings that must match between training and inference.
_META_KEYS = ("backbone", "input_size", "resize_mode", "mean", "std", "feature_dim")


@dataclass
class VerifyResult:
    is_true: bool
    score: float | None  # calibrated P(correct detection); None when the class has no SVM


class Verifier:
    def __init__(self, artifact: dict, device: str = "auto", unknown: str = "accept",
                 batch_size: int = 64):
        if unknown not in ("accept", "reject"):
            raise ValueError(f"unknown must be 'accept' or 'reject', got {unknown!r}")
        meta = artifact["meta"]
        self.extractor = FeatureExtractor(meta["backbone"], device=device,
                                          input_size=meta["input_size"], batch_size=batch_size)
        current = self.extractor.meta
        mismatched = {k: (meta.get(k), current[k]) for k in _META_KEYS if meta.get(k) != current[k]}
        if mismatched:
            raise ValueError(f"model was trained with different feature settings (saved, current): {mismatched}")
        for name, entry in artifact["classes"].items():
            n_in = entry["pipeline"].n_features_in_
            if n_in != self.extractor.feature_dim:
                raise ValueError(f"SVM for {name!r} expects {n_in}-d features, "
                                 f"{meta['backbone']} gives {self.extractor.feature_dim}")
        self.meta = meta
        self.pad = meta["pad"]
        self.classes: dict[str, dict] = artifact["classes"]
        self.unknown = unknown

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> Verifier:
        return cls(joblib.load(path), **kwargs)

    def verify(self, image: np.ndarray, boxes: Sequence[Sequence[float]],
               class_names: Sequence[Hashable], bgr: bool = False) -> list[VerifyResult]:
        """Verify detections on one frame. `boxes` are xyxy pixels; `image` is HxWx3 uint8."""
        if bgr:
            image = image[..., ::-1]
        crops = [crop_box(image, box, self.pad) for box in boxes]
        return self.verify_crops(crops, class_names)

    def verify_crops(self, crops: Sequence[np.ndarray],
                     class_names: Sequence[Hashable]) -> list[VerifyResult]:
        """Verify already-cropped RGB detections (cropped with crop_box and this model's pad)."""
        if len(crops) != len(class_names):
            raise ValueError(f"{len(crops)} crops but {len(class_names)} class names")
        default = VerifyResult(is_true=self.unknown == "accept", score=None)
        results = [default] * len(crops)
        known = [i for i, c in enumerate(class_names) if c in self.classes]
        if not known:
            return results

        feats = self.extractor.embed([crops[i] for i in known])  # one batched pass for all classes
        rows_by_class = defaultdict(list)
        for row, i in enumerate(known):
            rows_by_class[class_names[i]].append(row)
        for name, rows in rows_by_class.items():
            entry = self.classes[name]
            scores = entry["pipeline"].predict_proba(feats[rows])[:, 1]
            for row, score in zip(rows, scores):
                results[known[row]] = VerifyResult(bool(score >= entry["threshold"]), float(score))
        return results


def main():
    ap = argparse.ArgumentParser(description="Score a folder of crops with one class's SVM.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--crops", required=True, help="folder of crop images")
    ap.add_argument("--cls", required=True, help="class whose SVM to use")
    ap.add_argument("--out", help="CSV path (default: stdout)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    verifier = Verifier.load(args.model, device=args.device)
    if args.cls not in verifier.classes:
        ap.error(f"no SVM for class {args.cls!r}; available: {sorted(verifier.classes)}")
    paths = sorted(p for p in Path(args.crops).iterdir() if p.suffix.lower() in IMG_EXTS)

    out = open(args.out, "w", newline="") if args.out else sys.stdout
    writer = csv.writer(out)
    writer.writerow(["path", "score", "is_true"])
    n_true = 0
    for i in range(0, len(paths), 256):
        chunk = paths[i:i + 256]
        results = verifier.verify_crops([load_rgb(p) for p in chunk], [args.cls] * len(chunk))
        for path, r in zip(chunk, results):
            writer.writerow([path, f"{r.score:.4f}", r.is_true])
            n_true += r.is_true
    if out is not sys.stdout:
        out.close()
    threshold = verifier.classes[args.cls]["threshold"]
    print(f"{len(paths)} crops, {n_true} accepted as {args.cls!r} (threshold {threshold:.3f})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
