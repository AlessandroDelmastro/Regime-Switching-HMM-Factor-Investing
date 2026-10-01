# backtester.py
# Out-of-sample backtest of the regime-switching strategy. Every trading day from the start
# date the HMM is re-estimated on the previous `window_size` days, the regime of the last
# observed day is detected and, if the detection passes the confidence check, the portfolio
# switches to the strategy mapped to that regime. The return of the day is then the return
# of the strategy held.

import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import tqdm
from data_loader import fetch_data, calculate_features
from hmm_model import MarketHMM
from factor_analysis import fetch_ff_factors

def get_state_map(hmm, X_train):
    # Gives a name to each state of the fitted HMM. State numbers are arbitrary, so the
    # states are ordered by the mean return of the days assigned to them.
    states = hmm.predict(X_train)
    means = []
    vols = []
    
    for i in range(hmm.model.n_components):
        mask = (states == i)
        if mask.sum() > 0:
            means.append(X_train[mask, 0].mean()) # Return
            vols.append(X_train[mask, 1].mean())  # Volatility
        else:
            means.append(0)
            vols.append(0)
            
    sorted_indices = np.argsort(means)
    
    
    # Lowest mean return = Bear, highest = Bull; with 3 states the middle one is Neutral
    if len(sorted_indices) == 3:
        state_map = {
            sorted_indices[0]: 'Bear',
            sorted_indices[1]: 'Neutral',
            sorted_indices[2]: 'Bull'
        }
    else:
        state_map = {
            sorted_indices[0]: 'Bear',
            sorted_indices[1]: 'Bull'
        }
    
    
    return state_map

def calculate_pdf_confidence(hmm, state, obs):
    # Confidence check (as in Wang, Lin & Mikhelson, 2020): density of the observed return
    # and volatility under the normal distribution of the detected state. A high density
    # means the observation is typical of that regime.
    mean = hmm.model.means_[state]
    cov = hmm.model.covars_[state]
    
    # Extract params
    mu_ret = mean[0]
    mu_vol = mean[1]
    
    # Full covariance is used in model, but for 1D PDF check we take diagonal elements
    var_ret = cov[0, 0]
    var_vol = cov[1, 1]
    
    std_ret = np.sqrt(var_ret)
    std_vol = np.sqrt(var_vol)
    
    pdf_ret = stats.norm.pdf(obs[0], loc=mu_ret, scale=std_ret)
    pdf_vol = stats.norm.pdf(obs[1], loc=mu_vol, scale=std_vol)
    
    return pdf_ret, pdf_vol

def run_backtest(start_date='2018-01-01', window_size=2707, n_components=3):
    # start_date    first day of the out-of-sample test
    # window_size   number of trading days used to estimate the HMM each day (as in the paper)
    # n_components  number of regimes (2: Bull/Bear, 3: Bull/Neutral/Bear)
    
    print("Loading data...")
    
    # Download enough history before the start date to fill the first estimation window
    full_start_date = pd.to_datetime(start_date) - pd.Timedelta(days=window_size * 2) # Buffer
    df = fetch_data(start_date=full_start_date.strftime('%Y-%m-%d'))
    df = calculate_features(df).dropna()
    
    # Load Factors
    ff_data = fetch_ff_factors('2000-01-01', '2025-01-01')
    if ff_data.empty:
        print("Error: Could not fetch Factor data.")
        return

    # Handle whitespace in factor columns
    ff_data.columns = [c.strip() for c in ff_data.columns]
    
    # Align Data
    # (keep only the days present in both SPY and the factor data, matched by date)
    common_idx = df.index.intersection(ff_data.index)
    df = df.loc[common_idx]
    ff_data = ff_data.loc[common_idx]
    
    # Start Iteration
    test_dates = df.index[df.index >= pd.to_datetime(start_date)]
    if len(test_dates) == 0:
        print(f"No data found after {start_date}")
        return

    print(f"Starting backtest from {start_date} ({len(test_dates)} trading days)...")
    
    
    current_strategy = 'Market' # Start with Market
    portfolio_value = 100.0
    values = []
    decisions = []
    
    
    # Strategy held in each regime
    strat_map = {
        'Bull': 'Carhart',      
        'Neutral': 'AQR',      
        'Bear': 'Cash'          
    }
    
    # Position of the first test day in the data
    t_start = df.index.get_loc(test_dates[0])
    
    for t in tqdm(range(t_start, len(df))):
        date = df.index[t]
        
        train_start = t - window_size
        if train_start < 0:
            
            continue
            
        # Estimation window: the window_size days before day t (day t excluded)
        train_data = df.iloc[train_start:t]
        X_train = train_data[['Daily_Return', 'Volatility_10d_MSE']].values
    
        # New HMM estimated every day on the window
        hmm = MarketHMM(n_iter=20, n_components=n_components, random_state=42) # Lower iter for speed in loop
        try:
            hmm.model.fit(X_train)
        except:
            # Convergence failure fallback
            pass
            
        # 3. Identify Regimes
        state_labels = get_state_map(hmm, X_train)
        
        
        # Regime of the last observed day (t-1)
        last_obs = X_train[-1].reshape(1, -1) # Observation at t-1
        current_state_idx = hmm.model.predict(last_obs)[0]
        predicted_regime = state_labels[current_state_idx]
        
        pdf_ret, pdf_vol = calculate_pdf_confidence(hmm, current_state_idx, last_obs[0])
        
        # The regime is accepted only if both densities are above the thresholds of the paper
        confident = (pdf_vol > 0.3) and (pdf_ret > 0.5)
        
        # 6. Switching Logic
        if confident:
            target_strategy = strat_map[predicted_regime]
            if target_strategy != current_strategy:
                # Switch
                current_strategy = target_strategy
        else:
            # Keep previous strategy
            pass
            
        # Return of day t of the strategy held, from the factor returns of that day
        todays_factors = ff_data.loc[date]
        rf = todays_factors['RF']
        mkt = todays_factors['Mkt-RF']
        smb = todays_factors['SMB']
        hml = todays_factors['HML']
        hml = todays_factors['HML']
        mom = todays_factors['Mom']
        rmw = todays_factors['RMW']
        cma = todays_factors['CMA']
        
        if current_strategy == 'Market':
            day_ret = mkt + rf
        elif current_strategy == 'Fama-French':
            day_ret = mkt + smb + hml + rf
        elif current_strategy == 'Carhart':
            day_ret = mkt + smb + hml + mom + rf
        elif current_strategy == 'Value':
            day_ret =  mkt + hml + rf
        elif current_strategy == 'AQR':
            day_ret = mkt + hml + mom + rmw + cma + rf
        elif current_strategy == 'Cash':
            day_ret = rf
        else:
            day_ret = 0
            
        # Compound the portfolio value (starts at 100)
        portfolio_value *= (1 + day_ret)
        
        values.append({
            'Date': date,
            'Portfolio_Value': portfolio_value,
            'Strategy': current_strategy,
            'Regime': predicted_regime,
            'Confident': confident,
            'State_Idx': current_state_idx
        })
        
    
    results_df = pd.DataFrame(values).set_index('Date')
    

    # CAGR of the strategy and of SPY over the test period (calendar days)
    spy_start_price = df.loc[results_df.index[0], 'Close']
    spy_end_price = df.loc[results_df.index[-1], 'Close']
    spy_cagr = (spy_end_price / spy_start_price) ** (365 / (results_df.index[-1] - results_df.index[0]).days) - 1
    
    strat_end_value = results_df['Portfolio_Value'].iloc[-1]
    strat_cagr = (strat_end_value / 100.0) ** (365 / (results_df.index[-1] - results_df.index[0]).days) - 1
    
    print(f"\n--- Backtest Results (Jan 2018 - Present), {n_components} regimes ---")
    print(f"Dynamic Strategy CAGR: {strat_cagr*100:.2f}%")
    print(f"SPY Benchmark CAGR:    {spy_cagr*100:.2f}%")
    
    print("\nFinal Portfolio Value (Start=100):")
    print(f"Strategy: {strat_end_value:.2f}")
    
    print("\nStrategy Allocation Distribution:")
    print(results_df['Strategy'].value_counts(normalize=True))
    
    
    # Chart: strategy value against SPY rebased to 100
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    
   
    spy_curve = df.loc[results_df.index, 'Close']
    spy_curve = spy_curve / spy_curve.iloc[0] * 100
    
    plt.plot(results_df.index, results_df['Portfolio_Value'], label='HMM Strategy', linewidth=2)
    plt.plot(results_df.index, spy_curve, label='SPY Benchmark', linestyle='--', alpha=0.7)
    
    plt.title(f'HMM Dynamic Strategy vs SPY (2018-Present), {n_components} regimes')
    plt.ylabel('Portfolio Value (Start=100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'backtest_results_{n_components}_regimes.png')
    print(f"\nSaved plot to backtest_results_{n_components}_regimes.png")
    
    return results_df

if __name__ == "__main__":
    df_res = run_backtest()
