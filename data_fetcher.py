import requests
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
import os
import time

load_dotenv()
FMP_KEY = os.getenv("FMP_API_KEY")

# ─────────────────────────────────────────
# 1. PULL EARNINGS DATA FROM FMP
# ─────────────────────────────────────────

def get_earnings_history(ticker):
    url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{ticker}?apikey={FMP_KEY}"
    try:
        response = requests.get(url)
        data = response.json()

        if not data:
            print(f"No data for {ticker}")
            return None

        df = pd.DataFrame(data)
        df = df[['date', 'symbol', 'eps', 'epsEstimated', 'revenueEstimated', 'revenue']]
        df.rename(columns={
            'date': 'earnings_date',
            'symbol': 'ticker',
            'eps': 'actual_eps',
            'epsEstimated': 'estimated_eps'
        }, inplace=True)

        df.dropna(subset=['actual_eps', 'estimated_eps'], inplace=True)
        return df

    except Exception as e:
        print(f"Error fetching earnings for {ticker}: {e}")
        return None


# ─────────────────────────────────────────
# 2. CALCULATE SURPRISE % AND LABELS
# ─────────────────────────────────────────

def calculate_surprise(df):
    df['surprise_pct'] = (
        (df['actual_eps'] - df['estimated_eps']) / df['estimated_eps'].abs()
    ) * 100

    def assign_label(surprise):
        if surprise > 2:
            return 'Beat'
        elif surprise < -2:
            return 'Miss'
        else:
            return 'Meet'

    df['label'] = df['surprise_pct'].apply(assign_label)
    return df


# ─────────────────────────────────────────
# 3. PULL FUNDAMENTALS FROM YFINANCE
# ─────────────────────────────────────────

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        income = stock.quarterly_financials.T
        income.index = pd.to_datetime(income.index)

        income['revenue_growth'] = income['Total Revenue'].pct_change(periods=-1)
        income['gross_margin'] = income['Gross Profit'] / income['Total Revenue']

        income = income[['revenue_growth', 'gross_margin']].copy()
        income['ticker'] = ticker
        income.reset_index(inplace=True)
        income.rename(columns={'index': 'date'}, inplace=True)

        return income

    except Exception as e:
        print(f"Failed fundamentals for {ticker}: {e}")
        return None


# ─────────────────────────────────────────
# 4. MERGE EARNINGS + FUNDAMENTALS
# ─────────────────────────────────────────

def merge_datasets(earnings_df, fundamentals_df):
    earnings_df['earnings_date'] = pd.to_datetime(earnings_df['earnings_date'])
    fundamentals_df['date'] = pd.to_datetime(fundamentals_df['date'])

    merged_rows = []

    for _, row in earnings_df.iterrows():
        ticker = row['ticker']
        edate = row['earnings_date']

        fund_subset = fundamentals_df[fundamentals_df['ticker'] == ticker].copy()
        if fund_subset.empty:
            continue

        past = fund_subset[fund_subset['date'] <= edate]
        if past.empty:
            continue

        closest = past.loc[(past['date'] - edate).abs().idxmin()]

        merged_rows.append({
            'ticker': ticker,
            'earnings_date': edate,
            'actual_eps': row['actual_eps'],
            'estimated_eps': row['estimated_eps'],
            'surprise_pct': row['surprise_pct'],
            'label': row['label'],
            'revenue_growth': closest['revenue_growth'],
            'gross_margin': closest['gross_margin']
        })

    return pd.DataFrame(merged_rows)


# ─────────────────────────────────────────
# 5. CLEAN THE DATA
# ─────────────────────────────────────────

def clean_data(df):
    df.drop_duplicates(subset=['ticker', 'earnings_date'], inplace=True)
    df.dropna(inplace=True)
    df = df[df['surprise_pct'].between(-100, 100)]
    df.sort_values(['ticker', 'earnings_date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────
# 6. RUN EVERYTHING
# ─────────────────────────────────────────

if __name__ == "__main__":

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
               'NVDA', 'JPM', 'BAC', 'TSLA', 'WMT',
               'JNJ', 'PG', 'V', 'MA', 'HD']

    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/cleaned', exist_ok=True)

    # --- Pull earnings ---
    print("📥 Fetching earnings data...")
    all_earnings = []
    for ticker in tickers:
        print(f"  Fetching {ticker}...")
        df = get_earnings_history(ticker)
        if df is not None:
            all_earnings.append(df)
        time.sleep(0.3)

    if not all_earnings:
        print("❌ No earnings data fetched. Check your FMP API key in .env")
        exit()

    earnings_df = pd.concat(all_earnings, ignore_index=True)
    earnings_df = calculate_surprise(earnings_df)
    earnings_df.to_csv('data/raw/earnings_raw.csv', index=False)
    print(f"✅ Earnings data: {len(earnings_df)} rows")
    print(earnings_df['label'].value_counts())

    # --- Pull fundamentals ---
    print("\n📥 Fetching fundamentals...")
    all_fundamentals = []
    for ticker in tickers:
        print(f"  Fetching {ticker}...")
        df = get_fundamentals(ticker)
        if df is not None:
            all_fundamentals.append(df)
        time.sleep(0.5)

    if not all_fundamentals:
        print("❌ No fundamentals data fetched.")
        exit()

    fundamentals_df = pd.concat(all_fundamentals, ignore_index=True)
    fundamentals_df.to_csv('data/raw/fundamentals_raw.csv', index=False)
    print(f"✅ Fundamentals data: {len(fundamentals_df)} rows")

    # --- Merge ---
    print("\n🔗 Merging datasets...")
    merged_df = merge_datasets(earnings_df, fundamentals_df)

    # --- Clean ---
    print("🧹 Cleaning data...")
    final_df = clean_data(merged_df)

    # --- Save ---
    final_df.to_csv('data/cleaned/earnings_dataset.csv', index=False)
    print(f"\n✅ Done! Final dataset saved.")
    print(f"Shape: {final_df.shape}")
    print(f"\nLabel distribution:")
    print(final_df['label'].value_counts())
    print(f"\nPreview:")
    print(final_df.head())
