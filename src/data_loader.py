# data_loader.py
# Market data for the HMM: SPY daily prices and the two observed variables of the model
# (daily return and 10-day volatility).

import os
import yfinance as yf
import pandas as pd
import numpy as np

# Local copy of the SPY prices (saved from yfinance); if missing, data are downloaded
SPY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'spy_daily.csv')

def fetch_data(ticker="SPY", start_date="2007-01-01", end_date='2023-12-31'):
    # Daily prices of `ticker` between start_date and end_date (Open, High, Low, Close,
    # Volume). Close is adjusted for dividends and splits (yfinance default), so its
    # percentage change is the total return of the ETF.



    print(f"Fetching data for {ticker}...")
    if ticker == "SPY" and os.path.exists(SPY_FILE):
        # Use the local file when available, so every run works on the same data
        data = pd.read_csv(SPY_FILE, index_col=0, parse_dates=True)
        return data.loc[start_date:end_date]
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Ensure MultiIndex columns are handled if present (yfinance update)
    # (recent yfinance versions return columns like ('Close', 'SPY'): keep only 'Close')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    return data

def calculate_features(df):
    # Adds the two variables observed by the HMM. Both use only prices up to each day,
    # so the value of day t is known at the close of day t.



    df = df.copy()

    # 1. Daily Returns
    # r_t = Close_t / Close_{t-1} - 1
    df['Daily_Return'] = df['Close'].pct_change()

    # 2. Volatility (User defined: RMSE of returns relative to 10-day MA)
    # For each day: deviation of the return from its 10-day moving average, squared,
    # averaged over the last 10 days, square root. The first ~20 days are NaN
    # (two rolling windows) and are dropped later with dropna().

    ma_10 = df['Daily_Return'].rolling(window=10).mean()
    squared_diff = (df['Daily_Return'] - ma_10)**2
    df['Volatility_10d_MSE'] = np.sqrt(squared_diff.rolling(window=10).mean())

    return df

if __name__ == "__main__":
    # Quick check: download the data and print the first and last rows with the features
    # Fetch data
    df = fetch_data()

    if df.empty:
        print("No data fetched.")
    else:
        # Calculate features
        df = calculate_features(df).dropna()

        # Display output
        print(df.head())
        print(df.tail())

