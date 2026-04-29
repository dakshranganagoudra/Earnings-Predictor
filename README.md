# 📈 Earnings Surprise Predictor

A machine learning tool that predicts whether a company will **Beat**, **Meet**, or **Miss** Wall Street's earnings estimates — before the announcement.

Built with XGBoost, Financial Modeling Prep, Yahoo Finance, and Streamlit.

---

## 🚀 Setup

### 1. Clone and enter the project
```bash
git clone https://github.com/YOUR_USERNAME/earnings-predictor.git
cd earnings-predictor
```

### 2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys
```bash
cp .env.example .env
```
Then open `.env` and fill in:
```
FMP_API_KEY=your_key_here
```
Get a free key at https://financialmodelingprep.com

---

## ▶️ Running the Project

### Step 1 — Collect Data
```bash
python3 src/data_fetcher.py
```

### Step 2 — Build Features
```bash
python3 src/feature_engineering.py
```

### Step 3 — Train Model
```bash
python3 src/train_model.py
```

### Step 4 — Launch Dashboard
```bash
streamlit run app.py
```

---

## 🧠 How It Works

The model uses 16 features grouped into 5 categories:

| Category | Features |
|---|---|
| Surprise History | Previous surprise %, 4Q average, beat streak, consistency |
| Estimate Revisions | Analyst revision %, number of analysts |
| Fundamentals | Revenue growth, gross margin, acceleration, trend |
| Price Context | 30D return, 5D return, volatility, % from 52W high |
| Categorical | Fiscal quarter, ticker encoding |

---

## 📊 Tech Stack

- **Data:** Financial Modeling Prep API, Yahoo Finance (yfinance)
- **Modeling:** XGBoost, scikit-learn
- **UI:** Streamlit
- **Language:** Python 3.10+

---

## ⚠️ Disclaimer

This project is for educational purposes only and is not financial advice.
