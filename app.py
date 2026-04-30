import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib
import os
import time
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
FMP_KEY = os.getenv("FMP_API_KEY")

st.set_page_config(
    page_title="Earnings Surprise Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Earnings Surprise Predictor")
st.markdown("Predict whether a company will **Beat**, **Meet**, or **Miss** Wall Street's earnings estimates.")
st.divider()

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/xgb_earnings_predictor.pkl')
        le = joblib.load('models/label_encoder.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        imputer = joblib.load('models/imputer.pkl')
        return model, le, feature_cols, imputer
    except FileNotFoundError:
        return None, None, None, None

model, le, feature_cols, imputer = load_model()

if model is None:
    st.error("⚠️ Model not found. Please run `python3 train_model.py` first.")
    st.stop()

def get_earnings_history(ticker):
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings_history
        if earnings is None or earnings.empty:
            return None
        earnings = earnings.reset_index()
        earnings.rename(columns={
            'quarter': 'earnings_date',
            'epsActual': 'actual_eps',
            'epsEstimate': 'estimated_eps'
        }, inplace=True)
        earnings['ticker'] = ticker
        earnings['earnings_date'] = pd.to_datetime(earnings['earnings_date'])
        earnings = earnings[['earnings_date', 'actual_eps', 'estimated_eps']].copy()
        earnings.dropna(subset=['actual_eps', 'estimated_eps'], inplace=True)
        earnings['surprise_pct'] = ((earnings['actual_eps'] - earnings['estimated_eps']) / earnings['estimated_eps'].abs()) * 100
        return earnings.sort_values('earnings_date')
    except:
        return None

def get_price_features(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='3mo')
        if hist.empty:
            return {}
        close = hist['Close']
        return {
            'price_return_30d': float((close.iloc[-1] - close.iloc[-min(30, len(close))]) / close.iloc[-min(30, len(close))]),
            'price_return_5d': float((close.iloc[-1] - close.iloc[-min(5, len(close))]) / close.iloc[-min(5, len(close))]),
            'price_volatility_30d': float(close.pct_change().std()),
            'pct_from_52w_high': float((close.iloc[-1] - close.max()) / close.max())
        }
    except:
        return {}

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        income = stock.quarterly_financials.T
        if income.empty:
            return {}
        rev_col = next((c for c in income.columns if 'Revenue' in str(c) and 'Total' in str(c)), None)
        gp_col = next((c for c in income.columns if 'Gross' in str(c) and 'Profit' in str(c)), None)
        if rev_col is None or gp_col is None:
            return {}
        income['revenue_growth'] = income[rev_col].pct_change(fill_method=None, periods=-1)
        income['gross_margin'] = income[gp_col] / income[rev_col]
        latest = income.iloc[0]
        return {
            'revenue_growth': float(latest['revenue_growth']) if not pd.isna(latest['revenue_growth']) else 0.0,
            'gross_margin': float(latest['gross_margin']) if not pd.isna(latest['gross_margin']) else 0.0,
        }
    except:
        return {}

def build_features(ticker, history_df):
    features = {}

    if history_df is not None and len(history_df) >= 2:
        recent = history_df.tail(5)
        features['prev_surprise_1q'] = float(recent['surprise_pct'].iloc[-1]) if len(recent) > 0 else 0.0
        features['avg_surprise_4q'] = float(recent['surprise_pct'].mean())
        streak = 0
        for s in recent['surprise_pct'].values[::-1]:
            if s > 2:
                streak += 1
            else:
                break
        features['beat_streak'] = float(streak)
        features['surprise_consistency'] = float(recent['surprise_pct'].std()) if len(recent) > 1 else 0.0
    else:
        features.update({'prev_surprise_1q': 0.0, 'avg_surprise_4q': 0.0,
                         'beat_streak': 0.0, 'surprise_consistency': 5.0})

    features['eps_estimate_revision_pct'] = 0.0
    features['num_analysts'] = 5.0

    fund = get_fundamentals(ticker)
    features['revenue_growth'] = fund.get('revenue_growth', 0.0)
    features['gross_margin'] = fund.get('gross_margin', 0.3)
    features['revenue_growth_accel'] = 0.0
    features['gross_margin_trend'] = features['gross_margin']

    price = get_price_features(ticker)
    features['price_return_30d'] = price.get('price_return_30d', 0.0)
    features['price_return_5d'] = price.get('price_return_5d', 0.0)
    features['price_volatility_30d'] = price.get('price_volatility_30d', 0.02)
    features['pct_from_52w_high'] = price.get('pct_from_52w_high', -0.1)

    features['fiscal_quarter'] = float(datetime.now().month // 3 + 1)
    features['ticker_encoded'] = float(abs(hash(ticker)) % 500)

    return features

with st.sidebar:
    st.header("🔍 Make a Prediction")
    ticker_input = st.text_input("Enter Stock Ticker", value="AAPL", max_chars=10).upper().strip()
    predict_btn = st.button("🚀 Predict", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**About**")
    st.markdown("""
    This tool uses XGBoost trained on historical earnings data to predict whether a company will Beat, Meet, or Miss Wall Street estimates.
    
    **Data Sources:**
    - Yahoo Finance (earnings, price + fundamentals)
    """)

if predict_btn and ticker_input:
    with st.spinner(f"Analyzing {ticker_input}..."):

        col1, col2 = st.columns([1, 1])

        history_df = get_earnings_history(ticker_input)
        features = build_features(ticker_input, history_df)

        feature_vector = pd.DataFrame([[features.get(f, 0.0) for f in feature_cols]], columns=feature_cols)
        feature_vector_imputed = imputer.transform(feature_vector)

        pred_encoded = model.predict(feature_vector_imputed)[0]
        pred_proba = model.predict_proba(feature_vector_imputed)[0]
        pred_label = le.inverse_transform([pred_encoded])[0]
        confidence = float(max(pred_proba)) * 100

        with col1:
            st.subheader(f"📊 Prediction for {ticker_input}")
            color = {"Beat": "🟢", "Meet": "🟡", "Miss": "🔴"}.get(pred_label, "⚪")
            st.metric("Predicted Outcome", f"{color} {pred_label}")
            st.metric("Confidence", f"{confidence:.1f}%")
            st.markdown("**Probability Breakdown:**")
            for cls, prob in zip(le.classes_, pred_proba):
                st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

        with col2:
            if history_df is not None and not history_df.empty:
                st.subheader(f"📅 Recent Earnings History")
                display_df = history_df.tail(8)[['earnings_date', 'actual_eps', 'estimated_eps', 'surprise_pct']].copy()
                display_df['earnings_date'] = display_df['earnings_date'].dt.strftime('%Y-%m-%d')
                display_df['surprise_pct'] = display_df['surprise_pct'].round(2)
                display_df['actual_eps'] = display_df['actual_eps'].round(3)
                display_df['estimated_eps'] = display_df['estimated_eps'].round(3)
                display_df.columns = ['Date', 'Actual EPS', 'Est. EPS', 'Surprise %']
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                beats = (history_df['surprise_pct'] > 2).sum()
                total = len(history_df)
                beat_rate = beats / total * 100
                st.metric("Historical Beat Rate", f"{beat_rate:.0f}%", f"{beats}/{total} quarters")
            else:
                st.warning("Could not fetch earnings history for this ticker.")

    st.divider()
    st.subheader("🔑 Key Signals Used")
    sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
    with sig_col1:
        st.metric("Prev Quarter Surprise", f"{features['prev_surprise_1q']:.1f}%")
        st.metric("Avg Surprise (4Q)", f"{features['avg_surprise_4q']:.1f}%")
    with sig_col2:
        st.metric("Beat Streak", f"{int(features['beat_streak'])} quarters")
        st.metric("Gross Margin", f"{features['gross_margin']*100:.1f}%")
    with sig_col3:
        st.metric("30D Price Return", f"{features['price_return_30d']*100:.1f}%")
        st.metric("5D Price Return", f"{features['price_return_5d']*100:.1f}%")
    with sig_col4:
        st.metric("Price Volatility", f"{features['price_volatility_30d']*100:.2f}%")
        st.metric("% From 52W High", f"{features['pct_from_52w_high']*100:.1f}%")

st.divider()
if os.path.exists('outputs/feature_importance.png'):
    st.subheader("📐 What Drives Predictions?")
    st.image('outputs/feature_importance.png', use_container_width=True)

st.divider()
st.caption("Built with XGBoost and Yahoo Finance. For educational purposes only — not financial advice.")