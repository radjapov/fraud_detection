# 🚀 Fraud Detection Demo — IEEE-CIS + LGBM + SHAP  
### Cyber-Neo Edition UI · Blueprint Architecture · Docker-ready

<p align="center">
  <img src="https://img.shields.io/badge/Model-LGBM-%235b8cff?style=for-the-badge&logo=lightning" />
  <img src="https://img.shields.io/badge/SHAP-Enabled-%23ff4da6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Flask-Blueprint-%231572B6?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/Docker-Ready-%232496ED?style=for-the-badge&logo=docker" />
</p>

## ✨ Overview

This project demonstrates a full fraud‑detection pipeline built on:

- **IEEE‑CIS Fraud Detection dataset**
- **LightGBM classifier**
- **Feature engineering + preprocessing pipeline**
- **Threshold optimization**
- **SHAP explainability (via worker process)**
- **Modular Flask app using Blueprints**
- **Dark Cyber‑Neo UI**
- **Docker + Makefile + uv support**

The app allows you to:
- load & preprocess data  
- train a model  
- tune threshold  
- run predictions  
- visualize SHAP values  
- explore history  
- interact with a beautiful UI  

---

## 🧠 Project Architecture

```
fraud_detection/
│
├── fraud_app/
│   ├── __init__.py
│   ├── app.py
│   ├── api.py
│   ├── ui.py
│   ├── services.py
│   ├── shap_worker.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── main.js
│
├── artifacts/
│   ├── pipeline_ieee.joblib
│   ├── meta_ieee.joblib
│   ├── shap_explainer_ieee.joblib
│   └── meta_ieee_threshold.json
│
├── data/
│   └── ieee_prepared.csv
│
├── create_ieee_dataset.py
├── train_ieee_lgbm.py
├── choose_threshold_and_save.py
├── eval_threshold.py
│
├── Makefile
├── pyproject.toml
└── README.md
```

---

## ⚙️ Installation

### Option A — Standard Python
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option B — Using uv (recommended)
```
uv sync
```

---

## ▶️ Running the App

### Run via Flask entry point
```
python3 -m fraud_app
```

### Or using Makefile
```
make run-bp
```

---

## 🎯 Training the Model

1. Prepare dataset  
```
python create_ieee_dataset.py
```

2. Train LightGBM  
```
python train_ieee_lgbm.py
```

3. Evaluate and select threshold  
```
python choose_threshold_and_save.py
```

4. (Optional) Fine‑tune  
```
python eval_threshold.py
```

---

## 🔍 SHAP Explainability

SHAP is executed in **a worker process** to avoid segfaults common on macOS + LightGBM.

The app automatically:
- starts the worker  
- passes JSON‑encoded input  
- returns base value & feature contributions  
- renders them as a horizontal bar chart  

---

## 🖥 Cyber‑Neo UI

The UI includes:
- glowing buttons  
- smooth gradients  
- animated focus states  
- dark futuristic aesthetic  
- SHAP bar‑charts  
- history viewer  
- random fraud/normal sample generator  

---

## 🐳 Docker Support

Build the container:
```
docker build -t fraud-ui .
```

Run:
```
docker run -p 5001:5001 fraud-ui
```

---

## 🛠 Makefile Commands

```
make run            # run app_ui.py
make run-bp         # run blueprint version
make lint           # run isort + black check
make format         # autoformat code
make docker-build
make docker-run
```

---

## 📄 License

MIT — feel free to use in personal or commercial projects.

---

## 💬 Contact

ranvar26@gmail.com.  
For improvements — just ask.
