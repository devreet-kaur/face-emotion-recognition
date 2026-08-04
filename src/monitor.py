"""
src/monitor.py -- Abdul Raouf Zabalawi
EvidentlyAI drift detection: training reference distribution vs live inference logs.

    python src/monitor.py --make-reference    # build the reference CSV (run once)
    python src/monitor.py                     # generate the drift report
    python src/monitor.py --report            # same as above, explicit

Output: reports/drift/drift_report_YYYYMMDD_HHMMSS.html

WHY THIS FILE CHANGED FROM THE FIRST DRAFT
------------------------------------------
The draft could never run. It required two CSVs that nothing in the project produced:

    data/merged/train_reference_sample.csv   -- no stage wrote it
    data/inference_logs.csv                  -- src/app.py logged nothing

So `python src/monitor.py` raised FileNotFoundError on the first line of work, and the
Phase 2 checklist item "EvidentlyAI drift report generated" was unachievable. Two
changes close the loop:

  1. `--make-reference` computes the reference distribution from the training split
     itself, using the same deterministic split as src/dataset.py (random_state=42).
  2. src/app.py now appends one row of image statistics per detected face to
     data/inference_logs.csv, so the current-window data accumulates as the service
     is used.

It also dropped the unused `ClassificationPreset` import, which failed
`ruff check src/` in the CI lint job.

COLUMN COMPARABILITY
--------------------
face_width_ratio and face_height_ratio are meaningful only at inference time, where a
face is located inside a larger frame. The training images are already tight crops, so
those columns are written as 1.0 in the reference and are excluded from the drift
comparison rather than being silently compared against a constant. Drift is computed
over the columns that are genuinely comparable, and the exclusions are printed.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.report import Report

with open("params.yaml") as f:
    P = yaml.safe_load(f)

REF_PATH = Path(P["monitoring"]["reference_data_path"])
CUR_PATH = Path(P["monitoring"]["current_data_path"])
OUT_DIR = Path(P["monitoring"]["report_path"])
N_REFERENCE = P["monitoring"]["n_reference_samples"]

# Every column written by src/app.py's inference logger
FEATURE_COLS = [
    "mean_brightness",
    "std_brightness",
    "mean_contrast",
    "face_width_ratio",
    "face_height_ratio",
]

# Inference-only geometry: constant in the reference, so comparing it is meaningless
INFERENCE_ONLY_COLS = ["face_width_ratio", "face_height_ratio"]

LABEL_COL = "predicted_label"


def face_statistics(img_rgb: np.ndarray, width_ratio=1.0, height_ratio=1.0) -> dict:
    """
    The five features the drift monitor tracks. Kept identical to the version in
    src/app.py -- if these two ever disagree, every drift number is meaningless.

    mean_contrast is RMS contrast (std / mean), which is scale-invariant, so a
    uniformly brighter image does not register as higher contrast.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = float(gray.mean())
    return {
        "mean_brightness": round(mean / 255.0, 6),
        "std_brightness": round(float(gray.std()) / 255.0, 6),
        "mean_contrast": round(float(gray.std()) / (mean + 1e-6), 6),
        "face_width_ratio": round(float(width_ratio), 6),
        "face_height_ratio": round(float(height_ratio), 6),
    }


def make_reference() -> None:
    """
    Build the reference distribution from the training split.

    Uses src/dataset.py's build_splits() so the reference is drawn from exactly the
    images the model trained on -- the same deterministic split, random_state=42.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from dataset import build_splits
    except ImportError as exc:
        sys.exit(
            f"ERROR: could not import src/dataset.py ({exc}).\n"
            "Run this from the repo root: python src/monitor.py --make-reference"
        )

    print("Rebuilding the deterministic training split...")
    train_items, _, _ = build_splits()
    if not train_items:
        sys.exit(
            "ERROR: the training split is empty. Extract "
            "data/basic/Image/aligned.zip and confirm data/FER exists, then retry."
        )

    n = min(N_REFERENCE, len(train_items))
    rng = np.random.default_rng(42)
    picks = rng.choice(len(train_items), size=n, replace=False)
    print(f"Sampling {n} of {len(train_items)} training images "
          f"(n_reference_samples={N_REFERENCE})")

    rows = []
    for i in picks:
        img, _label, _src = train_items[int(i)]
        rows.append(face_statistics(img))

    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    REF_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REF_PATH, index=False)
    print(f"Wrote {REF_PATH}  ({len(df)} rows)")
    print("\nReference means:")
    for col in FEATURE_COLS:
        print(f"  {col}: {df[col].mean():.4f}")
    print(
        f"\nNote: {', '.join(INFERENCE_ONLY_COLS)} are written as 1.0 because training "
        "images are already tight crops. They are excluded from drift comparison."
    )


def load_csv(path: Path, what: str, hint: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: {what} not found at {path}\n{hint}")
    df = pd.read_csv(path)
    if df.empty:
        sys.exit(f"ERROR: {path} exists but has no rows.\n{hint}")
    return df


def comparable_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list:
    """Columns present in both frames, minus the documented inference-only geometry."""
    shared = [c for c in FEATURE_COLS if c in reference.columns and c in current.columns]
    usable = [c for c in shared if c not in INFERENCE_ONLY_COLS]
    skipped = [c for c in shared if c in INFERENCE_ONLY_COLS]
    missing = [c for c in FEATURE_COLS if c not in shared]
    if skipped:
        print(f"  excluded (inference-only geometry): {', '.join(skipped)}")
    if missing:
        print(f"  excluded (absent from one side):    {', '.join(missing)}")
    if not usable:
        sys.exit(
            "ERROR: no comparable feature columns. Confirm data/inference_logs.csv "
            f"has the headers written by src/app.py: {', '.join(FEATURE_COLS)}"
        )
    return usable


def summarize_drift(reference: pd.DataFrame, current: pd.DataFrame, cols: list) -> None:
    print("\nDistribution comparison (training reference vs live inference):")
    for col in cols:
        ref_mean = float(reference[col].mean())
        cur_mean = float(current[col].mean())
        shift = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-8) * 100
        flag = "  <-- large shift" if shift > 20 else ""
        print(f"  {col}: ref={ref_mean:.4f}  current={cur_mean:.4f}  "
              f"shift={shift:5.1f}%{flag}")

    if LABEL_COL in reference.columns and LABEL_COL in current.columns:
        ref_dist = reference[LABEL_COL].value_counts(normalize=True)
        cur_dist = current[LABEL_COL].value_counts(normalize=True)
        print("\nPredicted-label distribution shift:")
        for label in sorted(set(ref_dist.index) | set(cur_dist.index)):
            print(f"  {label}: ref={ref_dist.get(label, 0):.3f}  "
                  f"current={cur_dist.get(label, 0):.3f}")
    elif LABEL_COL in current.columns:
        cur_dist = current[LABEL_COL].value_counts(normalize=True)
        print("\nPredicted-label distribution (inference only, no reference labels):")
        for label, share in cur_dist.items():
            print(f"  {label}: {share:.3f}")


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                     cols: list) -> str:
    metrics = [DataDriftPreset(), DatasetDriftMetric()]
    metrics += [ColumnDriftMetric(column_name=c) for c in cols]

    report = Report(metrics=metrics)
    report.run(reference_data=reference[cols], current_data=current[cols])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"drift_report_{stamp}.html"
    report.save_html(str(out_path))
    print(f"\nDrift report saved: {out_path}")
    print("Open it in a browser -- the summary panel reports how many columns drifted.")
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(
        description="EvidentlyAI drift detection for the emotion recognition service"
    )
    ap.add_argument("--make-reference", action="store_true",
                    help="build the reference CSV from the training split, then exit")
    ap.add_argument("--report", action="store_true",
                    help="generate the drift report (default action)")
    args = ap.parse_args()

    if args.make_reference:
        make_reference()
        return

    print("Loading reference distribution...")
    reference = load_csv(
        REF_PATH, "reference data",
        "Build it first:  python src/monitor.py --make-reference",
    )
    print(f"  {len(reference)} reference samples from the training distribution")

    print("Loading live inference logs...")
    current = load_csv(
        CUR_PATH, "inference logs",
        "src/app.py appends a row per detected face. Start the service and send a few\n"
        "requests first:\n"
        "  uvicorn src.app:app --port 5001\n"
        '  curl -X POST http://localhost:5001/predict -F "file=@some_face.jpg"',
    )
    print(f"  {len(current)} logged inference samples")

    print("\nSelecting comparable columns:")
    cols = comparable_columns(reference, current)
    print(f"  comparing: {', '.join(cols)}")

    summarize_drift(reference, current, cols)
    run_drift_report(reference, current, cols)


if __name__ == "__main__":
    main()
