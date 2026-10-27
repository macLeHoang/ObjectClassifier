"""Train one binary SVM per object class on frozen CNN features of detector crops.

Expected layout (crops made with feature_extractor.crop_box):
    <data>/<class>/true/*.jpg    correct detections of <class>
    <data>/<class>/false/*.jpg   false positives the detector labelled <class>

Name files "<group><sep><anything>" (e.g. "vid03__f0120_d2.jpg") so crops from the
same video/scene stay on one side of the train/val split.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from feature_extractor import IMG_EXTS, BACKBONES, FeatureExtractor, load_rgb, preprocess

C_GRID = [0.01, 0.1, 1.0, 10.0]
C_PARAM = "calibratedclassifiercv__estimator__C"

log = logging.getLogger("train_svm")


class CropDataset(Dataset):
    def __init__(self, paths: list[Path], size: int):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return preprocess(load_rgb(self.paths[i]), self.size)


def build_pipeline(C: float = 1.0) -> Pipeline:
    # L2-normalize raw CNN features, standardize, then a linear SVM with Platt-scaled probabilities.
    svm = LinearSVC(C=C, class_weight="balanced", max_iter=10000)
    return make_pipeline(Normalizer(), StandardScaler(),
                         CalibratedClassifierCV(svm, method="sigmoid", cv=3))


def scan_class(class_dir: Path) -> tuple[list[Path], np.ndarray]:
    paths, labels = [], []
    for sub, label in (("true", 1), ("false", 0)):
        d = class_dir / sub
        if d.is_dir():
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
            paths += files
            labels += [label] * len(files)
    return paths, np.array(labels, dtype=np.int64)


def file_groups(paths: list[Path], sep: str, name: str) -> np.ndarray:
    """Group id = filename prefix before `sep`; files without `sep` are their own group."""
    no_sep = sum(sep not in p.name for p in paths)
    if no_sep == len(paths):
        log.warning("[%s] no filename contains %r: splitting per crop, val metrics may be "
                    "optimistic if crops come from the same video/scene", name, sep)
    elif no_sep:
        log.warning("[%s] %d/%d filenames lack %r and are treated as their own group",
                    name, no_sep, len(paths), sep)
    return np.array([p.name.split(sep, 1)[0] if sep in p.name else str(p) for p in paths])


def embed_paths(extractor: FeatureExtractor, paths: list[Path], workers: int) -> np.ndarray:
    loader = DataLoader(CropDataset(paths, extractor.input_size), batch_size=extractor.batch_size,
                        num_workers=workers, pin_memory=extractor.device.type == "cuda")
    return np.concatenate([extractor.embed_batch(b) for b in tqdm(loader, desc="embed", unit="batch")])


def class_features(extractor: FeatureExtractor, paths: list[Path], cache_file: Path,
                   workers: int, name: str) -> np.ndarray:
    """Features for `paths`, re-embedding only files that are new or changed since the last run."""
    keys = [f"{p.resolve()}:{p.stat().st_mtime_ns}" for p in paths]
    cached = {}
    if cache_file.exists():
        with np.load(cache_file) as z:
            cached = dict(zip(z["keys"].tolist(), z["feats"]))
    missing = [i for i, k in enumerate(keys) if k not in cached]
    log.info("[%s] features: %d cached, %d to embed", name, len(keys) - len(missing), len(missing))
    if missing:
        new = embed_paths(extractor, [paths[i] for i in missing], workers)
        cached.update(zip((keys[i] for i in missing), new))
    feats = np.stack([cached[k] for k in keys]).astype(np.float32)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, keys=np.array(keys), feats=feats)
    return feats


def pick_val_split(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    """First grouped fold whose train and val sides both contain true and false crops."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in sgkf.split(np.zeros(len(y)), y, groups):
        if len(np.unique(y[tr])) == 2 and len(np.unique(y[va])) == 2:
            return tr, va
    return None


def pick_threshold(y: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """Highest score threshold that still accepts >= target_recall of the true crops."""
    _, recall, thresholds = precision_recall_curve(y, scores)
    ok = np.flatnonzero(recall[:-1] >= target_recall)  # recall[0] == 1, so never empty
    return float(thresholds[ok[-1]])


def val_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel())
    return {
        "val_ap": float(average_precision_score(y, scores)),
        "val_precision": tp / max(tp + fp, 1),
        "val_recall": tp / max(tp + fn, 1),
        "val_fp_rejection": tn / max(tn + fp, 1),
        "val_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def train_class(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                args: argparse.Namespace) -> dict | None:
    n_splits = max(2, round(1 / args.val_frac))
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        log.warning("[%s] only %d groups, need >= %d for a grouped val split; skipping",
                    name, n_groups, n_splits)
        return None
    split = pick_val_split(y, groups, n_splits, args.seed)
    if split is None:
        log.warning("[%s] no grouped split has both labels on each side; skipping", name)
        return None
    tr, va = split
    n_train_groups = len(np.unique(groups[tr]))
    log.info("[%s] split: train %d crops / %d groups, val %d crops / %d groups",
             name, len(tr), n_train_groups, len(va), len(np.unique(groups[va])))

    inner_splits = min(3, n_train_groups)
    if inner_splits >= 2:
        grid = GridSearchCV(build_pipeline(), {C_PARAM: C_GRID}, scoring="average_precision",
                            cv=StratifiedGroupKFold(n_splits=inner_splits), n_jobs=-1)
        grid.fit(X[tr], y[tr], groups=groups[tr])
        best_C, cv_ap, model = grid.best_params_[C_PARAM], float(grid.best_score_), grid.best_estimator_
    else:
        log.warning("[%s] too few train groups for C search; using C=1", name)
        best_C, cv_ap, model = 1.0, None, build_pipeline().fit(X[tr], y[tr])

    scores = model.predict_proba(X[va])[:, 1]
    threshold = pick_threshold(y[va], scores, args.target_recall)
    metrics = {
        "n_true": int(y.sum()), "n_false": int(len(y) - y.sum()),
        "n_train": len(tr), "n_val": len(va), "C": best_C, "cv_ap": cv_ap, "threshold": threshold,
        **val_metrics(y[va], scores, threshold),
    }
    log.info("[%s] C=%g val_AP=%.3f threshold=%.3f precision=%.3f recall=%.3f fp_rejected=%.1f%%",
             name, best_C, metrics["val_ap"], threshold, metrics["val_precision"],
             metrics["val_recall"], 100 * metrics["val_fp_rejection"])

    # Final model sees all of this class's data; the threshold comes from the held-out val split.
    final = build_pipeline(best_C).fit(X, y)
    return {"pipeline": final, "threshold": threshold, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="root folder: <class>/{true,false}/*.jpg")
    ap.add_argument("--out", default="models/verifier.joblib")
    ap.add_argument("--backbone", default="resnet18", choices=sorted(BACKBONES))
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--pad", type=float, default=0.15,
                    help="crop_box pad the training crops were made with; recorded for inference")
    ap.add_argument("--target-recall", type=float, default=0.98,
                    help="fraction of true detections the threshold must keep (on val)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--group-sep", default="__")
    ap.add_argument("--min-per-label", type=int, default=10)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    class_dirs = sorted(d for d in Path(args.data).iterdir() if d.is_dir())
    if not class_dirs:
        sys.exit(f"no class folders in {args.data}")

    extractor = FeatureExtractor(args.backbone, device=args.device,
                                 input_size=args.input_size, batch_size=args.batch_size)
    log.info("backbone %s (%d-d) on %s", args.backbone, extractor.feature_dim, extractor.device)
    cache_root = Path(args.cache_dir) / f"{args.backbone}_{args.input_size}"

    classes = {}
    for class_dir in class_dirs:
        name = class_dir.name
        paths, y = scan_class(class_dir)
        n_true, n_false = int(y.sum()), int(len(y) - y.sum())
        if min(n_true, n_false) < args.min_per_label:
            log.warning("[%s] %d true / %d false crops, need >= %d of each; skipping",
                        name, n_true, n_false, args.min_per_label)
            continue
        X = class_features(extractor, paths, cache_root / f"{name}.npz", args.workers, name)
        entry = train_class(name, X, y, file_groups(paths, args.group_sep, name), args)
        if entry is not None:
            classes[name] = entry
    if not classes:
        sys.exit("no class could be trained")

    meta = extractor.meta | {"pad": args.pad, "sklearn": sklearn.__version__,
                             "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"meta": meta, "classes": classes}, out)
    report = out.with_name(f"{out.stem}_report.json")
    report.write_text(json.dumps({"meta": meta, "classes": {k: v["metrics"] for k, v in classes.items()}},
                                 indent=2))
    log.info("saved %d class model(s) to %s (report: %s)", len(classes), out, report)


if __name__ == "__main__":
    main()
