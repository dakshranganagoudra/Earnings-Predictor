import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
import requests
import os
import time

load_dotenv()
FMP_KEY = os.getenv("FMP_API_KEY")


# ─────────────────────────────────────────
# FEATURE GROUP 1 — SURPRISE HISTORY
# ─────────────────────────────────────────

def add_surprise_history(df):
    df = df.sort_values(['ticker', 'earnings_date']).copy()

    df['prev_surprise_1q'] = df.groupby('ticker')['surprise_pct'].shift(1)

    df['avg_surprise_4q'] = (
        df.groupby('ticker')['surprise_pct']
        .transform(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
    )

    def calc_streak(series):
        streaks = []
        streak = 0
        for val in series:
            if pd.isna(val):
                streaks.append(np.nan)
                streak = 0
            elif val > 2:
                streak += 1
                streaks.append(streak)
            else:
                streak = 0
                streaks.append(streak)
        return streaks

    df['beat_streak'] = df.groupby('ticker')['surprise_pct'].transform(
        lambda x: pd.Series(calc_streak(x.shift(1).values), index=x.index)
    )

    df['surprise_consistency'] = (
        df.groupby('ticker')['surprise_pct']
        .transform(lambda x: x.shift(1).rolling(4, min_periods=2).std())
    )

    return df


# ─────────────────────────────────────────
# FEATURE GROUP 2 — ESTIMATE REVISIONS
# ─────────────────────────────────────────

def get_estimate_revisions(ticker, earnings_date):
    url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{ticker}?apikey={FMP_KEY}&limit=20"
    try:
        response = requests.get(url)
        data = response.json()

        if not data:
            return None

        est_df = pd.DataFrame(data)
        est_df['date'] = pd.to_datetime(est_df['date'])

        window_start = earnings_date - pd.Timedelta(days=30)
        window = est_df[
            (est_df['date'] >= window_start) &
            (est_df['date'] < earnings_date)
        ]

        if window.empty:
            return None

        latest = window.sort_values('date').iloc[-1]
        earliest = window.sort_values('date').iloc[0]

        denom = abs(earliest.get('estimatedEpsAvg', 1)) or 1
        revision = {
            'eps_estimate_latest': latest.get('estimatedEpsAvg', np.nan),
            'eps_estimate_revision_pct': (
                (latest.get('estimatedEpsAvg', np.nan) - earliest.get('estimatedEpsAvg', np.nan))
                / denom
            ) * 100,
            'num_analysts': latest.get('numberAnalystEstimatedEps', np.nan)
        }
        return revision

    except Exception as e:
        return None


def add_revision_features(df):
    revisions = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 20 == 0:
            print(f"  Revisions: {i}/{total}")
        rev = get_estimate_revisions(row['ticker'], row['earnings_date'])
        if rev:
            revisions.append(rev)
        else:
            revisions.append({
                'eps_estimate_latest': np.nan,
                'eps_estimate_revision_pct': np.nan,
                'num_analysts': np.nan
            })
        time.sleep(0.3)

    rev_df = pd.DataFrame(revisions)
    df = pd.concat([df.reset_index(drop=True), rev_df], axis=1)
    return df


# ─────────────────────────────────────────
# FEATURE GROUP 3 — FUNDAMENTAL MOMENTUM
# ─────────────────────────────────────────

def add_fundamental_features(df):
    df = df.sort_values(['ticker', 'earnings_date']).copy()

    df['revenue_growth_accel'] = df.groupby('ticker')['revenue_growth'].diff()

    df['gross_margin_trend'] = (
        df.groupby('ticker')['gross_margin']
        .transform(lambda x: x.shift(1).rolling(2, min_periods=2).mean())
    )

    df['revenue_growth_yoy'] = (
        df.groupby('ticker')['revenue_growth']
        .transform(lambda x: x.shift(4))
    )

    return df


# ─────────────────────────────────────────
# FEATURE GROUP 4 — PRICE CONTEXT
# ─────────────────────────────────────────

def get_price_features(ticker, earnings_date):
    try:
        stock = yf.Ticker(ticker)
        start = earnings_date - pd.Timedelta(days=60)
        end = earnings_date - pd.Timedelta(days=1)

        hist = stock.history(start=start, end=end)

        if hist.empty or len(hist) < 5:
            return None

        close = hist['Close']

        return {
            'price_return_30d': (close.iloc[-1] - close.iloc[-min(30, len(close))]) / close.iloc[-min(30, len(close))],
            'price_return_5d': (close.iloc[-1] - close.iloc[-min(5, len(close))]) / close.iloc[-min(5, len(close))],
            'price_volatility_30d': close.pct_change().std(),
            'pct_from_52w_high': (close.iloc[-1] - close.max()) / close.max()
        }

    except Exception as e:
        return None


def add_price_features(df):
    price_features = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 20 == 0:
            print(f"  Price features: {i}/{total}")
        pf = get_price_features(row['ticker'], row['earnings_date'])
        if pf:
            price_features.append(pf)
        else:
            price_features.append({
                'price_return_30d': np.nan,
                'price_return_5d': np.nan,
                'price_volatility_30d': np.nan,
                'pct_from_52w_high': np.nan
            })
        time.sleep(0.2)

    price_df = pd.DataFrame(price_features)
    df = pd.concat([df.reset_index(drop=True), price_df], axis=1)
    return df


# ─────────────────────────────────────────
# FEATURE GROUP 5 — CATEGORICAL
# ─────────────────────────────────────────

def add_categorical_features(df):
    df['fiscal_quarter'] = df['earnings_date'].dt.quarter
    df['month'] = df['earnings_date'].dt.month
    df['ticker_encoded'] = df['ticker'].astype('category').cat.codes
    return df


# ─────────────────────────────────────────
# BUILD FINAL FEATURE MATRIX
# ─────────────────────────────────────────

def build_feature_matrix(df):
    feature_cols = [
        'prev_surprise_1q', 'avg_surprise_4q', 'beat_streak', 'surprise_consistency',
        'eps_estimate_revision_pct', 'num_analysts',
        'revenue_growth', 'gross_margin', 'revenue_growth_accel', 'gross_margin_trend',
        'price_return_30d', 'price_return_5d', 'price_volatility_30d', 'pct_from_52w_high',
        'fiscal_quarter', 'ticker_encoded'
    ]

    model_df = df[feature_cols + ['label', 'ticker', 'earnings_date']].copy()

    for col in feature_cols:
        model_df[col] = model_df[col].fillna(model_df[col].median())

    model_df.dropna(subset=['label'], inplace=True)
    model_df.reset_index(drop=True, inplace=True)

    return model_df, feature_cols


# ─────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("📥 Loading cleaned earnings dataset...")
    df = pd.read_csv('data/cleaned/earnings_dataset.csv', parse_dates=['earnings_date'])
    print(f"Loaded {len(df)} rows")

    print("\n🔧 Adding surprise history features...")
    df = add_surprise_history(df)

    print("🔧 Adding estimate revision features (this takes a few minutes)...")
    df = add_revision_features(df)

    print("🔧 Adding fundamental momentum features...")
    df = add_fundamental_features(df)

    print("🔧 Adding price context features (this takes a few minutes)...")
    df = add_price_features(df)

    print("🔧 Adding categorical features...")
    df = add_categorical_features(df)

    print("\n📐 Building final feature matrix...")
    model_df, feature_cols = build_feature_matrix(df)

    os.makedirs('data/cleaned', exist_ok=True)
    model_df.to_csv('data/cleaned/feature_matrix.csv', index=False)

    print(f"\n✅ Feature matrix saved!")
    print(f"Shape: {model_df.shape}")
    print(f"Features: {feature_cols}")
    print(f"\nLabel distribution:")
    print(model_df['label'].value_counts())
    print(f"\nPreview:")
    print(model_df.head())
