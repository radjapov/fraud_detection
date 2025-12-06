<div align="center">

<img src="docs/neo_fraud_logo.png" alt="FraudDetect Neo Logo" width="180"/>

# FraudDetect: Transaction Risk Scoring with SHAP in a Dark Cyber‑Neo UI

**IEEE-CIS Fraud Detection → LightGBM pipeline → SHAP explanations → Interactive Flask UI**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-68bc00)](https://github.com/microsoft/LightGBM)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-00bcd4)](#-license)

</div>

---

## 🔮 Project overview

This repo implements **FraudDetect: a transaction risk scoring system** built on top of the
**IEEE-CIS Fraud Detection** dataset. The model is trained with **LightGBM**, wrapped
in a **sklearn `Pipeline`**, and exposed through a **Flask web UI** with a **Dark Cyber‑Neo**
visual style.

Key features:

- 🧠 **ML pipeline**: feature engineering + preprocessing + LightGBM classifier
- 🎯 **Risk scoring**: calibrated fraud probability with configurable threshold
- 🧩 **SHAP explainability**: local feature attribution for each prediction
- 🖥️ **Modern UI**: Dark Cyber‑Neo themed single-page app (vanilla JS + Canvas)
- 📊 **Model metrics panel**: ROC‑AUC, PR‑AUC, confusion matrix, F1 @ threshold
- 📂 **Artifacts-driven design**: pipeline + meta + SHAP explainer + metrics JSON

---

## 🧱 Architecture

High‑level blocks:

```text
         ┌──────────────────────────────────────────┐
         │              IEEE raw CSVs              │
         │  train_transaction.csv, train_identity  │
         └──────────────────────────────────────────┘
                             │
                             ▼
                 create_ieee_dataset.py
              (feature engineering + labels)
                             │
                             ▼
         ┌──────────────────────────────────────────┐
         │           data/ieee_prepared.csv        │
         └──────────────────────────────────────────┘
                             │
                             ▼
                  train_ieee_lgbm.py
     (sklearn ColumnTransformer + LightGBM pipeline,
       evaluation, SHAP explainer build, artifacts)
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                           artifacts/                            │
│   ├─ pipeline_ieee.joblib          # sklearn Pipeline           │
│   ├─ meta_ieee.joblib              # feature list + meta        │
│   ├─ shap_explainer_ieee.joblib    # SHAP TreeExplainer         │
│   ├─ meta_ieee_threshold.json      # chosen threshold τ         │
│   └─ metrics_ieee.json             # global + threshold metrics │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Flask app (fraud_app)                   │
│   ├─ app.py             # create_app(), blueprint registration  │
│   ├─ api.py             # /api/predict, /api/shap, /api/metrics │
│   ├─ services.py        # artifacts, SHAP worker, history       │
│   └─ shap_worker.py     # separate SHAP process                 │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Dark Cyber‑Neo UI                         │
│   templates/ui.html  +  static/style.css  +  static/main.js     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Main components

### 1. Dataset preparation

Script: **`create_ieee_dataset.py`**

- Reads `data/ieee/train_transaction.csv` and (optionally) `train_identity.csv`
- Builds engineered features such as:
  - `amount`
  - `hour` (derived from `TransactionDT`)
  - `card_age_months`
  - `sender_txn_24h`
  - `sender_avg_amount`
  - `distance_km`
  - `ip_risk`
  - `receiver_new`
  - `device_new`
  - `is_foreign`
  - `mcc`
  - `country_risk`
  - `is_fraud` (label)
- Writes compact training file:

```bash
python3 create_ieee_dataset.py \
  --input-dir data/ieee \
  --output data/ieee_prepared.csv
```

Output: **`data/ieee_prepared.csv`** (13 columns: 12 features + `is_fraud`).

---

### 2. Model training + SHAP explainer

Script: **`train_ieee_lgbm.py`**

This script:

1. Loads `data/ieee_prepared.csv`
2. Splits into train/val/test
3. Builds an sklearn `Pipeline`:
   - `ColumnTransformer` with:
     - numeric pipeline: `SimpleImputer(median)` + `StandardScaler`
     - categorical pipeline: `SimpleImputer("NA")` + `OneHotEncoder`
   - `LightGBM` classifier (`LGBMClassifier`)
4. Trains with early stopping (robust to different LightGBM versions)
5. Computes evaluation metrics on the test set
6. Saves artifacts to `artifacts/`:
   - `pipeline_ieee.joblib`
   - `meta_ieee.joblib`
   - `train_report_ieee.json`
   - `shap_explainer_ieee.joblib` (TreeExplainer, if SHAP is available)

Example:

```bash
python3 train_ieee_lgbm.py \
  --input data/ieee_prepared.csv \
  --out-dir artifacts \
  --sample 0.5   # optional speed-up
```

---

### 3. Threshold search

Script: **`choose_threshold_and_save.py`**

- Computes `predict_proba` on **full prepared dataset**
- Scans thresholds from 0.00 to 0.99 in small steps
- Logs metrics for each threshold
- Finds the threshold **τ\*** that maximizes F1
- Persists it into `artifacts/meta_ieee_threshold.json`

Example usage:

```bash
python3 choose_threshold_and_save.py \
  --data data/ieee_prepared.csv \
  --pipeline artifacts/pipeline_ieee.joblib \
  --out-json artifacts/meta_ieee_threshold.json
```

---

### 4. Global metrics export

Script: **`eval_model_metrics.py`**

- Uses the trained pipeline and full prepared dataset
- Computes:
  - Global **ROC‑AUC**
  - Global **PR‑AUC**
  - Metrics at current threshold τ (from `meta_ieee_threshold.json`):
    - accuracy, precision, recall, F1
    - confusion matrix
- Saves everything into **`artifacts/metrics_ieee.json`**

Example:

```bash
python3 eval_model_metrics.py \
  --data data/ieee_prepared.csv \
  --pipeline artifacts/pipeline_ieee.joblib \
  --threshold-json artifacts/meta_ieee_threshold.json \
  --out-json artifacts/metrics_ieee.json
```

This JSON is later consumed by the UI via `/api/metrics`.

---

### 5. Flask app with Blueprints

Package: **`fraud_app/`**

- `app.py` – `create_app()`, registers blueprints, serves `ui.html`
- `api.py` – REST endpoints:
  - `GET  /api/random?fraud=0|1`
  - `POST /api/predict`
  - `POST /api/shap`
  - `GET  /api/history`
  - `GET  /api/metrics`
- `services.py` – all heavy lifting:
  - load artifacts (pipeline, meta, SHAP, thresholds, metrics)
  - feature normalization for incoming JSON
  - prediction via pipeline
  - SHAP calculation via worker
  - prediction history storage
- `shap_worker.py` – spawned as a separate process for SHAP:
  - reads a single row from stdin
  - loads SHAP explainer + pipeline artifacts
  - outputs SHAP values as JSON

Run in blueprint mode:

```bash
python3 -m fraud_app
```

By default the app listens on `http://0.0.0.0:5001`.

---

### 6. Dark Cyber‑Neo UI

- Template: `templates/ui.html`
- Styles: `static/style.css`
- Logic: `static/main.js`

The UI provides:

- Form for 12 engineered features
- Buttons:
  - **Random normal** – sample a non‑fraud transaction from dataset
  - **Random fraud** – sample a fraud transaction
  - **Clear** – reset form
  - **Predict** – call `/api/predict`
  - **Explain (SHAP)** – call `/api/shap`
- Prediction block:
  - Big label: **High fraud risk** (red neon) / **Low fraud risk** (green neon)
  - Probability and threshold
- SHAP block:
  - Centered **horizontal bar chart** in Canvas
  - Top‑8 features by |SHAP|
  - Positive contributions in red, negative in blue
  - Compact list under the chart
- History block:
  - Toggle button
  - List of recent predictions (timestamp + probability + label)
- Metrics block:
  - Samples, fraud rate
  - ROC‑AUC, PR‑AUC
  - Accuracy@τ, F1@τ
  - Confusion matrix and threshold

All styled in a **Dark Cyber‑Neo** theme: glowing borders, blur, neon accents.

---

## 📊 Current model metrics

These values are loaded from `artifacts/metrics_ieee.json` and displayed in the UI.

Below are example metrics (your actual values may differ):

```
Samples: 590540
Fraud rate: 0.0349

ROC-AUC: 0.9752
PR-AUC: 0.8043

Threshold τ = 0.35

Accuracy: 0.96016
F1: 0.35068

Confusion matrix:
TN: 560661
FP: 9216
FN: 14310
TP: 6353
```

You can recompute these at any time by rerunning:

```bash
python3 eval_model_metrics.py ...
```

and then refreshing the UI.

---

## 🛠 Local setup

### 1. Clone the repo

```bash
git clone git@github.com:radjapov/fraud_detection.git
cd fraud_detection
```

### 2. Prepare data (optional, if you don’t already have artifacts)

Download the IEEE‑CIS Fraud Detection CSVs from Kaggle and place them in:

```text
data/ieee/train_transaction.csv
data/ieee/train_identity.csv   # optional but recommended
```

Then:

```bash
python3 create_ieee_dataset.py
python3 train_ieee_lgbm.py
python3 choose_threshold_and_save.py
python3 eval_model_metrics.py
```

If you already have filled `artifacts/`, you can skip this.

### 3. Run Flask UI

```bash
python3 -m fraud_app
```

Open in browser:

```text
http://127.0.0.1:5001
```

---

## 🧪 Development tooling

- **isort** – import sorting
- **black** – code formatting
- **uv** – fast Python package/dependency manager (optional)
- **Makefile** – shortcuts for common commands

Examples:

```bash
make lint     # isort + black --check
make format   # isort + black (in-place)
make run-bp   # python3 -m fraud_app
```

> If you use `uv`, you can wrap commands like `uv run python train_ieee_lgbm.py` etc.

---

## 🧬 Ideas for future work

- Add **Docker** setup with a pinned NumPy / SHAP stack
- Expose **batch scoring API** with CSV/Parquet upload
- Integrate **authentication** and role‑based access
- Add more **domain‑specific features** (velocity, merchant graph, device fingerprinting)
- Plug in **stream processing** (Kafka / Pulsar) for near real‑time scoring

---

## 📄 License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

Feel free to fork, experiment and adapt it for your own fraud‑detection experiments.
