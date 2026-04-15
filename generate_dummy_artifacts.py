import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Note: We skip generating a real SHAP explainer here because pickling a local class
# causes issues when loading it in a different process/context.
# For testing, we mock the SHAP functionality.

def generate_dummy_artifacts():
    # Create dummy data
    X = pd.DataFrame({
        "amount": np.random.rand(100) * 100,
        "hour": np.random.randint(0, 24, 100),
        "card_age_months": np.random.randint(0, 60, 100),
        "sender_txn_24h": np.random.randint(0, 10, 100),
        "sender_avg_amount": np.random.rand(100) * 50,
        "distance_km": np.random.rand(100) * 10,
        "ip_risk": np.random.rand(100),
        "mcc": np.random.choice([4812, 5411, 5912, 6011, 7995], 100),
        "country_risk": np.random.choice([0, 1, 2], 100),
        "receiver_new": np.random.choice([0, 1], 100),
        "device_new": np.random.choice([0, 1], 100),
        "is_foreign": np.random.choice([0, 1], 100),
    })
    y = np.random.randint(0, 2, 100)

    # Train a dummy pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())
    ])
    pipe.fit(X, y)

    # Save pipeline
    joblib.dump(pipe, ARTIFACTS_DIR / "pipeline_ieee.joblib")

    # Save meta
    meta = {
        "features": list(X.columns),
        "target": "is_fraud",
        "n_train": 100,
        "n_val": 20,
        "n_test": 20,
        "chosen_threshold": 0.5
    }
    joblib.dump(meta, ARTIFACTS_DIR / "meta_ieee.joblib")

    # Save metrics
    metrics = {
        "roc_auc": 0.8,
        "pr_auc": 0.6,
        "classification_report": {},
        "confusion_matrix": [[40, 10], [10, 40]]
    }
    with open(ARTIFACTS_DIR / "metrics_ieee.json", "w") as f:
        json.dump(metrics, f)

    # Save threshold
    with open(ARTIFACTS_DIR / "meta_ieee_threshold.json", "w") as f:
        json.dump({"threshold": 0.5}, f)

    # Create dummy dataset for random sampling
    X["is_fraud"] = y
    X.to_csv("data/ieee_prepared.csv", index=False)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_dummy_artifacts()
    print("Dummy artifacts generated.")
