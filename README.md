<div align="center">

# FraudDetect: Transaction Risk Scoring with SHAP in a Dark Cyber‑Neo UI

**IEEE-CIS Fraud Detection → LightGBM pipeline → SHAP explanations → Interactive Flask UI**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-68bc00)](https://github.com/microsoft/LightGBM)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-00bcd4)](#-license)

</div>

---

## 🔮 Project overview

**FraudDetect** scores card transactions from the **IEEE-CIS Fraud Detection** dataset. A **LightGBM**
model inside a **scikit-learn `Pipeline`** is served by a **Flask** app with a **Dark Cyber‑Neo** UI
that explains each prediction with **SHAP**.

- 🧠 **ML pipeline**: 130 features → preprocessing → LightGBM, trained and evaluated on a **chronological split with late labels**
- 🎯 **Risk scoring**: fraud probability + a decision threshold chosen on validation data that behaves like the future
- 🧩 **SHAP explainability**: per-transaction attribution, folded back to the input features
- 🖥️ **UI**: the input form is generated from the model's own feature schema (`GET /api/schema`)
- 📊 **Metrics panel**: ROC‑AUC, PR‑AUC, confusion matrix, F1 @ threshold on the test period, plus the validation F1
- 📂 **Artifacts-driven**: pipeline + meta + SHAP explainer + threshold + metrics JSON

---

## 📊 Results (chronological test split, 30-day label delay)

The data is cut by time (`TransactionDT`). The **newest 20% of transactions (118 108 rows, 3.4%
fraud) is the test set**, the "future". Chargebacks are reported weeks after a transaction, so a
live model does not know the labels of recent transactions yet: everything the model learns from is
**at least 30 days older than the test period** (`--label-delay-days`, default 30). Training is
two-stage:

1. **Stage A**: fit on the oldest labeled rows, then early-stop and pick the threshold τ on a
   **validation block that is separated from the training rows by the same 30-day gap**. Its scores
   look like scores of future transactions, so validation F1 is an honest forecast of test F1.
2. **Stage B**: refit on **all** labeled rows (the freshest data, validation block included) with
   the tree count found in stage A, scaled to the larger training set.

Test data was not used to choose features, hyperparameters, τ or the delay.

| Same LightGBM, same protocol | Features | ROC-AUC | PR-AUC* | F1 @ τ, test | F1 @ τ, validation |
|---|---|---|---|---|---|
| 10 original proxy features | 10 | 0.798 | 0.244 | 0.301 | 0.280 |
| + our own features (card, email, identity flag, amount decimals, weekday) | 23 | 0.831 | 0.299 | 0.342 | 0.333 |
| + Vesta counters `C`, `D`, `M` | 61 | 0.894 | 0.484 | 0.479 | 0.470 |
| **Production: + 30 Vesta `V` columns and the identity block** | **130** | **0.901** | **0.506** | **0.483** | **0.492** |

\*PR-AUC is average precision; each row is the mean of two seeds (noise ≈ ±0.003 ROC-AUC, ±0.006 F1).
The shipped model (seed 42) scores ROC-AUC 0.900, PR-AUC 0.508. Threshold τ = 0.2325 (max F1 on
validation, 0.493):

```
Test rows: 118108   (fraud: 4064)

Precision: 0.508   Recall: 0.468   F1: 0.487   Accuracy: 0.966

TN: 112199   FP: 1845
FN:  2161    TP: 1903
```

### Does F1 still "drift" between validation and test?

| Protocol | F1 on validation | F1 on test | Gap |
|---|---|---|---|
| Before: validation glued to the training rows, model not refit (61 features) | 0.519 | 0.461 | 0.058 |
| **Now: validation behind a 30-day gap, refit on all labeled rows** | 0.493 (95% CI 0.478–0.507) | 0.487 (95% CI 0.473–0.500) | **0.006** |

The two intervals overlap completely; before, they did not. Three things were suspected:

- **The threshold**: no. Using the validation threshold on the test period costs 0.005 F1 against
  the best possible threshold (about 0.002–0.006 in the backtest windows).
- **An over-optimistic validation block**: yes, this was the culprit. A block right behind the
  training rows contains the same cards in the same weeks the model just saw. In a backtest inside
  the pre-test data it scored F1 0.59 against 0.49 for the following weeks (gap 0.10); with the
  label gap the estimate is 0.48 against 0.45.
- **Period difficulty**: partly, and it does not go away. At one fixed threshold the weekly F1 of
  the test period runs 0.41, 0.50, 0.47, 0.47, 0.56, 0.51. Expect weeks like these, not a constant.

The change also improved the model itself. Paired bootstrap on the same test rows, against the
previous model: **F1 +0.025 (95% CI 0.016–0.036)**, PR-AUC +0.049 (0.043–0.056), ROC-AUC 0.881 → 0.900.
About four fifths of the F1 gain comes from the protocol (refit on fresher and more data, threshold
from a gap-separated block: 0.461 → 0.479 with the same 61 features); the new `V` / identity
features add +0.022 PR-AUC but only +0.005 F1, which is within the noise. Predicted probabilities
are also better calibrated: mean 0.0355 against a real fraud rate of 0.0344 (0.82 of it before).

Other models, same split, features and protocol (`compare_baselines.py`; `xgboost` is skipped
unless installed):

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| **LightGBM, production settings** | **0.900** | **0.508** | **0.487** |
| LightGBM (500 trees, class-balanced, no early stopping) | 0.890 | 0.492 | 0.452 |
| RandomForest | 0.856 | 0.433 | 0.374 |
| LogisticRegression | 0.826 | 0.184 | 0.324 |

### What the label delay costs

Production features, mean of two seeds. The delay is used both for the gap before the test period
and for the gap in front of the validation block.

| Label delay | ROC-AUC | PR-AUC | F1, validation | F1, test | Gap |
|---|---|---|---|---|---|
| 0 days | 0.921 | 0.573 | 0.589 | 0.551 | +0.038 |
| 7 days | 0.915 | 0.553 | 0.560 | 0.532 | +0.028 |
| 14 days | 0.909 | 0.536 | 0.549 | 0.518 | +0.030 |
| **30 days** | **0.901** | **0.506** | **0.492** | **0.483** | **+0.009** |

A 30-day delay costs about 0.02 ROC-AUC and 0.07 F1 against a world where labels arrive at once.
IEEE-CIS calls a transaction legitimate only after 120 quiet days, but a 60-day delay cannot be
evaluated with this protocol on six months of data: the gap-separated validation block would leave
no training rows in front of it. Plan for the 30-day row, not the top one.

### Do the identifier-like features earn their place?

`card2/3/5` and `addr1/2` let a model recognise the cards and addresses it saw fraud on. IEEE-CIS
also marks later transactions of a charged-back card as fraud, so this is easier here than live.
Test metrics with and without them (mean of two seeds):

| Label delay | Full model: ROC / PR / F1 | Without ID-like: ROC / PR / F1 | Gain from ID-like: ROC / PR |
|---|---|---|---|
| 0 days | 0.921 / 0.573 / 0.551 | 0.916 / 0.554 / 0.533 | +0.005 / +0.018 |
| 30 days | 0.901 / 0.506 / 0.483 | 0.894 / 0.494 / 0.471 | +0.007 / +0.012 |

With a 30-day delay they still give a small, consistent gain, so they stay; treat it as
partly optimistic (the labelling rule propagates fraud to a card's later transactions).

### How to read these numbers

- **Most of the gain comes from Vesta's masked columns.** `C1–C14`, `D1–D15`, `M1–M9` and the `V`
  columns are pre-computed by the data provider and their windows are undocumented. A check found no
  sign that `C` looks into the future (partial correlation with a client's *future* activity ≤ 0.08
  once the past is fixed), but that cannot be proven. Without them the model is at ROC-AUC 0.83.
- **One fixed delay is a simplification.** Real reports arrive anywhere between days and four
  months after the transaction; the split cuts everything at exactly 30 days.
- **A single F1 is a noisy number.** The bootstrap interval of the test F1 is ±0.013, and different
  weeks differ by more (see above).
- **Random split, leaky features and no time order overstate quality.** The first version scored
  ROC-AUC 0.887 that way; the ten original proxy features are worth 0.80 on a chronological split
  with a delay.
- Several original feature names are proxies, not real fields (see the feature table below).

<details>
<summary>What was tried, and what did not help (validation / backtest inside the pre-test data)</summary>

**Feature groups added to the original 12 (validation, single LightGBM, no delay).**

| Added | ROC-AUC | PR-AUC | Decision |
|---|---|---|---|
| — (12 original) | 0.817 | 0.303 | |
| `amount_decimal`, `dow` | 0.831 | 0.319 | kept |
| card fields + `addr1/2` | 0.831 | 0.339 | kept (`card1`, a pure ID, dropped) |
| email domains | 0.834 | 0.351 | kept |
| `has_identity`, `DeviceType` | 0.818 | 0.309 | kept (helps together with the rest) |
| `dist1`, `dist2` | 0.819 | 0.308 | dropped, no gain |
| `M1–M9` | 0.840 | 0.320 | kept |
| `D1–D15` | 0.865 | 0.420 | kept |
| `C1–C14` | 0.887 | 0.522 | kept |
| causal client history (8 `uid_*` features) | 0.828 | 0.321 | dropped: nothing on top of `C/D` |
| all together | 0.924 | 0.601 | |

**Levers for F1 (two 28-day backtest windows inside the pre-test data, two seeds, F1 at the
validation threshold).** The best possible F1 with a perfect threshold was 0.497–0.504 for every
lever, so only new signal or fresher data moves it. Each block has its own baseline: the first
uses the production preprocessing, the second a LightGBM with native categoricals.

| Lever | F1 | PR-AUC | Verdict |
|---|---|---|---|
| baseline A (61 features) | 0.499 | 0.506 | |
| newer samples weigh more (half-life 30 / 60 days) | 0.495 / 0.497 | 0.505 / 0.503 | no gain |
| drop features that drift (`card_age_months`, …) | 0.498 – 0.501 | 0.511 | PR-AUC +0.005, F1 unchanged |
| refit on train + validation, threshold from the glued validation | 0.491 | 0.515 | ranking better, threshold no longer fits |
| baseline B (61 features) | 0.494 | 0.500 | |
| gap-separated validation + refit (production protocol) | 0.501 | 0.516 | kept |
| + all 339 `V` columns | 0.498 | 0.511 | unstable between windows |
| + 30 `V` columns chosen on training rows (`select_v_features.py`) | 0.505 | 0.512 | kept |
| + identity block | 0.500 | 0.506 | kept |
| + 30 `V` + identity | 0.508 | 0.516 | kept |

Dropping `uid_*`, `receiver_new` / `device_new`, `card1` and `dist1/2` each changed validation
ROC-AUC by less than the seed noise (±0.002), so they were dropped. `card1` also cost 0.007
PR-AUC, but a raw card identifier lets a model memorise individual cards, so it stays out.

</details>

---

## 🧱 Architecture

```text
         ┌──────────────────────────────────────────┐
         │              IEEE raw CSVs              │
         │  train_transaction.csv, train_identity  │
         └──────────────────────────────────────────┘
                             │
                             ▼
                 create_ieee_dataset.py
        (features from fraud_app/schema.py, causal sender
         history, chronological order, TransactionDT kept)
                             │
                             ▼
         ┌──────────────────────────────────────────┐
         │           data/ieee_prepared.csv        │
         └──────────────────────────────────────────┘
                             │  chronological split + label-delay gaps (features.py)
                             ▼
                  train_ieee_lgbm.py
    stage A: fit → early stop + validation scores (gap-separated)
    stage B: refit on all labeled rows, SHAP explainer, artifacts
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                           artifacts/                            │
│   ├─ pipeline_ieee.joblib          # sklearn Pipeline (refit)   │
│   ├─ meta_ieee.joblib              # feature list + protocol    │
│   ├─ shap_explainer_ieee.joblib    # SHAP TreeExplainer         │
│   ├─ val_scores_ieee.npz           # stage-A validation scores  │
│   ├─ meta_ieee_threshold.json      # τ, chosen on those scores  │
│   └─ metrics_ieee.json             # test + validation metrics  │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Flask app (fraud_app)                   │
│   ├─ app.py / ui.py     # create_app(), blueprints              │
│   ├─ api.py             # /api/* endpoints                      │
│   ├─ services.py        # artifacts, history, form schema       │
│   ├─ features.py        # input typing, preprocessing, split    │
│   ├─ schema.py          # feature groups, labels, types         │
│   ├─ explain.py         # SHAP on the preprocessed matrix       │
│   └─ shap_worker.py     # SHAP in a separate process            │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Dark Cyber‑Neo UI                         │
│   templates/index.html  +  static/style.css  +  static/main.js  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧬 Features (130)

Defined once in [`fraud_app/schema.py`](fraud_app/schema.py); the trainer, the dataset builder and the
UI all read it.

| Group | Features | Notes |
|---|---|---|
| Transaction (5) | `amount`, `amount_decimal`, `hour`, `dow`, `mcc` | `mcc` is `ProductCD` coded 0–4 (no real MCC in the data); `dow` is a weekday proxy with unknown offset |
| Sender history (3) | `card_age_months`, `sender_txn_24h`, `sender_avg_amount` | causal: only transactions up to the current one, sender ≈ `card1` |
| Card & billing (7) | `card2`, `card3`, `card4`, `card5`, `card6`, `addr1`, `addr2` | masked attributes; `card4/6` are categorical |
| Email & device (4) | `P_emaildomain`, `R_emaildomain`, `has_identity`, `DeviceType` | rare categories (< 500 training rows) share one bucket |
| Location & risk (4) | `distance_km`, `ip_risk`, `country_risk`, `is_foreign` | proxies: `ip_risk` = scaled mean of `V1..V50`, `country_risk` = `addr1` bucket, `is_foreign` = `dist1 > 500` or no `addr1` |
| Vesta counters (14) | `C1`–`C14` | masked, provider-computed |
| Vesta time deltas (15) | `D1`–`D15` | masked, empty = unknown |
| Vesta match flags (9) | `M1`–`M9` | `T`/`F`/`M0–M2`, empty = unknown |
| Vesta engineered (30) | `V13`, `V20`, … `V323` | the 30 most useful of `V1–V339`, picked on training rows only (`select_v_features.py`) |
| Identity & device (39) | `id_01`–`id_38`, `DeviceInfo` | present for about a quarter of the transactions; `id_30/31/33` and `DeviceInfo` are OS, browser, screen size, device model |

Missing values stay missing: numeric `NaN` goes straight into LightGBM, categorical gaps become the
label `NA`. Categories are encoded by one function (`fraud_app.features.category_labels`) in training
**and** serving; the encoder learns strings such as `"4"`, so sending `4.0` would otherwise be an
unseen category and silently become all zeros.

---

## 📦 Main components

### 1. Dataset preparation — `create_ieee_dataset.py`

Reads `data/ieee/train_transaction.csv` (+ optional `train_identity.csv`), sorts by time, builds
the features above and writes **`data/ieee_prepared.csv`** (130 features + `is_fraud` +
`TransactionDT`; the time column is only used to split and is never a model feature).

```bash
python3 create_ieee_dataset.py --input-dir data/ieee --output data/ieee_prepared.csv
```

`select_v_features.py` picks the `V` columns (importance from the training rows of the production
split only); paste its output into `V_SELECTED` in the schema, then rebuild the dataset.

### 2. Training — `train_ieee_lgbm.py`

1. Splits the data as described under *Results*: test = newest 20%, labeled rows end
   `--label-delay-days` (default 30) before it, validation block behind a gap of the same size.
2. `ColumnTransformer`: numeric → `StandardScaler` (NaN passes through), categorical →
   `OneHotEncoder` (rare categories pooled).
3. **Stage A**: `LGBMClassifier` (63 leaves, `min_child_samples 80`, `reg_lambda 5`,
   `colsample 0.4`, `subsample 0.8`) with early stopping on validation AUC; the validation
   scores are saved to `val_scores_ieee.npz`.
4. **Stage B**: preprocessing and classifier refit on every labeled row with the stage-A tree
   count scaled to the larger set. This is the served model.
5. Saves `pipeline_ieee.joblib`, `meta_ieee.joblib`, `train_report_ieee.json` and a SHAP
   `TreeExplainer` (100 background rows keep each explanation fast).

```bash
python3 train_ieee_lgbm.py --input data/ieee_prepared.csv --out-dir artifacts [--label-delay-days 30]
```

The delay is stored in `meta_ieee.joblib`; the metrics and baseline scripts read it from there, so
they always evaluate on the split the model was trained with.

### 3. Threshold — `choose_treshold_and_save.py`

The served model has seen the validation rows, so its own scores on them mean nothing. The script
picks the F1-optimal τ from the **stage-A validation scores** in `val_scores_ieee.npz` and writes
`artifacts/meta_ieee_threshold.json`. The test period is never used.

### 4. Metrics — `eval_model_metrics.py`

Scores the **test** split with the served model and writes `artifacts/metrics_ieee.json`: ROC-AUC,
PR-AUC, precision / recall / F1 / confusion matrix at τ, the label delay, and the stage-A
`validation` block for comparison. The UI reads it via `/api/metrics`.

```bash
python3 choose_treshold_and_save.py && python3 eval_model_metrics.py
```

### 5. Baselines — `compare_baselines.py`

Trains LightGBM, RandomForest, LogisticRegression (and XGBoost if installed) with the same split
(`--label-delay-days` defaults to the production model's), preprocessing and two-stage protocol;
writes `artifacts/baseline_comparison.json`.

### 6. Flask app — `fraud_app/`

| Endpoint | |
|---|---|
| `GET /api/schema` | grouped feature description the UI builds its form from |
| `GET /api/random?fraud=0\|1` | a random transaction from the **held-out test period** |
| `POST /api/predict` | probability + label; send any subset of features, the rest are "missing" |
| `POST /api/shap` | SHAP per input feature (separate process, 15 s timeout, ~1–2 s in practice) |
| `GET /api/history` | last 200 predictions |
| `GET` / `POST /api/threshold` | read / change τ (POST: see below) |
| `GET /api/metrics` | contents of `metrics_ieee.json` |

```bash
python3 -m fraud_app          # http://127.0.0.1:5001
```

| Environment variable | Default | Purpose |
|---|---|---|
| `FRAUD_HOST` | `127.0.0.1` | bind address (the Docker image sets `0.0.0.0`) |
| `FRAUD_PORT` | `5001` | port |
| `FRAUD_ADMIN_TOKEN` | unset | if set, `POST /api/threshold` needs header `X-Admin-Token`; if unset, only requests from localhost may change τ |

### 7. UI — `templates/index.html`, `static/`

- The form is built at load time from `/api/schema`: ten grouped, collapsible sections, categorical
  fields as drop-downs filled with the categories the model actually keeps (rare ones are pooled).
  **Leave a field empty when the value is unknown.**
- **Random normal / Random fraud** fill the form from a held-out transaction, **Predict** and
  **Explain (SHAP)** call the API, **Clear** resets.
- Prediction block (label, probability, τ), top-8 SHAP bar chart with a compact list, history
  toggle, and the metrics panel (test-split figures, the label gap and the validation F1).

---

## 🛠 Local setup

```bash
git clone git@github.com:radjapov/fraud_detection.git
cd fraud_detection
uv sync --extra dev                    # exact versions from uv.lock
# without uv:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
```

Model artifacts are pickles, so `scikit-learn`, `lightgbm` and `shap` must match the versions that
trained them (`requirements.txt` pins them).

Download the IEEE‑CIS CSVs from Kaggle into `data/ieee/` (`train_transaction.csv`, and
`train_identity.csv` for the identity block), then:

```bash
python3 create_ieee_dataset.py
python3 train_ieee_lgbm.py
python3 choose_treshold_and_save.py
python3 eval_model_metrics.py
python3 -m fraud_app
```

Without the raw data, `python3 generate_dummy_artifacts.py` creates a tiny stand-in model and
dataset so the app and the tests can run.

### Docker

`docker compose up --build` builds the image and mounts `./artifacts` and `./data`; set
`FRAUD_ADMIN_TOKEN` in `docker-compose.yml` to change τ over HTTP.

---

## 🧪 Development

```bash
pytest -q           # 34 tests: API, feature typing, missing values, SHAP, time split + label gaps, two-stage training, causality
make lint           # isort + black --check
make format         # isort + black
```

The tests never touch your real `artifacts/meta_ieee_threshold.json` or `history.json`.

---

## 🧬 Ideas for future work

- A per-transaction label delay (report dates) instead of one fixed cutoff.
- Rolling re-training and threshold re-tuning on recent data.
- Batch scoring API (CSV / Parquet upload).
- Real per-client history from a feature store instead of masked provider counters.

---

## 📄 License

MIT. See [`LICENSE`](LICENSE).
