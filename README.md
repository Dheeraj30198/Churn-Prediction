# 📉 Churn Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-prediction-system300.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> An AI-powered customer churn prediction system built for subscription-based businesses.  
> Upload your customer CSV → get churn predictions, risk levels, and reasons — instantly.

---

## 🌐 Live Demo

**👉 [Try it live on Streamlit Cloud](https://churn-prediction.streamlit.app)**

No setup needed — just upload a CSV and explore predictions in your browser!

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔮 **Churn Prediction** | Predicts whether each customer will churn |
| 📊 **Risk Level** | Classifies customers as Low / Medium / High risk |
| 💬 **Churn Reason** | Predicts the likely reason for churn |
| 📈 **Interactive Charts** | 6 Plotly charts with hover, zoom & drill-down |
| ⚙️ **Adjustable Threshold** | Tune prediction sensitivity via sidebar slider |
| ⬇️ **Export Results** | Download full prediction CSV |
| 🌙 **Dark UI** | Premium dark theme with glassmorphism effects |

---

## 🏗️ Architecture

```
Churn-Prediction/
├── streamlit_app.py           # 🚪 Streamlit Cloud entry point
├── churn_model.py             # 🤖 ML training pipeline (EDA + modeling + tuning)
├── reason_model.py            # 💬 Churn-reason model training
├── predict.py                 # 🔮 Batch prediction script
├── recommendations.py         # 💡 Retention recommendation engine
├── requirements.txt           # 📦 Python dependencies
├── Telco_customer_churn.xlsx  # 📊 Training dataset
├── sample_input_6_customers.csv  # 🧪 Sample data for testing
│
├── app/                       # 📱 Streamlit dashboard package
│   ├── dashboard.py           # Main UI (premium dark theme)
│   ├── visualization.py       # Plotly interactive charts
│   ├── utils.py               # Prediction & data utilities
│   └── config.py              # Paths & constants
│
├── artifacts/                 # 🗂️ Trained models & reports
│   ├── models/
│   │   ├── churn_model.joblib
│   │   └── reason_model.joblib
│   └── reports/
│       ├── metrics.json
│       └── feature_importance.csv
│
└── .streamlit/
    └── config.toml            # 🎨 Dark theme configuration
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repo
```bash
git clone https://github.com/Dheeraj30198/Churn-Prediction.git
cd Churn-Prediction
```

### 2. Create virtual environment
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run the dashboard
```powershell
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) and upload `sample_input_6_customers.csv` to test!

---

## 🧠 Train Your Own Model

### Step 1 — Train the churn model
```powershell
.\.venv\Scripts\python churn_model.py --data-path Telco_customer_churn.xlsx --target-col "Churn Value" --artifact-dir artifacts
```

**Outputs:**
- `artifacts/eda/` — EDA charts, summary, missing-values table
- `artifacts/models/churn_model.joblib` — trained model
- `artifacts/reports/metrics.json` — accuracy, F1, ROC-AUC
- `artifacts/reports/feature_importance.csv`

### Step 2 — Train the churn-reason model
```powershell
.\.venv\Scripts\python reason_model.py --data-path Telco_customer_churn.xlsx --artifact-dir artifacts
```

---

## 📦 Dataset

Uses the **Telco Customer Churn** dataset as a proxy for subscription-based businesses (Netflix, SaaS, telecom, etc.).

| Field | Type | Description |
|-------|------|-------------|
| Tenure Months | Numeric | How long the customer has been with the service |
| Monthly Charges | Numeric | Monthly billing amount |
| Total Charges | Numeric | Total amount charged |
| Churn Value | Binary | 1 = churned, 0 = stayed |
| Churn Reason | Categorical | Reason for churning |
| ... | ... | 30+ other customer features |

---

## 🤝 Contributing

Contributions are welcome! This is a collaborative project.

1. Fork the repo (or push directly if you're a collaborator)
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add: your feature"`
4. Push and open a Pull Request

---

## 👥 Authors

- **Dheeraj** — [@Dheeraj30198](https://github.com/Dheeraj30198) — Core ML pipeline
- **Contributors** — See [GitHub contributors](https://github.com/Dheeraj30198/Churn-Prediction/graphs/contributors)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
