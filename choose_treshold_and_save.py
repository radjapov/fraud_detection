#!/usr/bin/env python3
# choose_treshold_and_save.py — F1-optimal decision threshold, saved for the app.
#
# The served model was refit on the validation rows, so its own scores on them are in-sample.
# The threshold is therefore chosen from the scores the stage-A model (trained without the
# validation block) gave them; train_ieee_lgbm.py stores those in artifacts/val_scores_ieee.npz.
# The validation block is separated from the training rows by the label-delay gap, so these
# scores behave like scores of future transactions. The test period is never used here.

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
VAL_SCORES = ART / "val_scores_ieee.npz"
THR_JSON = ART / "meta_ieee_threshold.json"


def best_threshold(y, probs, steps=401):
    """Threshold with the highest F1 on (y, probs); returns (threshold, f1)."""
    best_thr, best_f1 = 0.0, -1.0
    for t in np.linspace(0, 1, steps):
        f1 = f1_score(y, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr, best_f1


def main():
    print("Running choose_treshold_and_save.py")

    if not VAL_SCORES.exists():
        raise FileNotFoundError(f"No validation scores: {VAL_SCORES} (run train_ieee_lgbm.py)")

    data = np.load(VAL_SCORES)
    y, probs = data["y"].astype(int), data["p"].astype(float)
    best_thr, best_f1 = best_threshold(y, probs)

    print(f"Best val F1={best_f1:.4f} at threshold={best_thr:.4f} (n_val={len(y)})")

    with open(THR_JSON, "w") as f:
        json.dump(
            {"threshold": best_thr, "chosen_metric": "f1", "split": "val", "val_f1": best_f1}, f
        )

    print("Saved:", THR_JSON)


if __name__ == "__main__":
    main()
