# factor_analysis.py
# Fama-French factors and performance of the factor strategies inside each HMM regime.
# The results show which strategy works best in each regime, i.e. what the portfolio
# should hold when the HMM detects that regime.

import os
import pandas as pd
import numpy as np
from data_loader import fetch_data, calculate_features
from hmm_model import MarketHMM

# Local copies of the Kenneth French daily files (same files pandas_datareader downloads)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
FF5_FILE = os.path.join(DATA_DIR, 'F-F_Research_Data_5_Factors_2x3_daily.csv')
MOM_FILE = os.path.join(DATA_DIR, 'F-F_Momentum_Factor_daily.csv')


def read_french_csv(path):
    """Reads a daily CSV of the French data library (values in %, dates YYYYMMDD)."""
    # The file starts with a few lines of description, then a header line
    # (",Mkt-RF,SMB,...") just before the first row of data, and ends with a copyright line
    lines = open(path, encoding='latin-1').read().splitlines()
    is_day = lambda l: l.split(',')[0].strip().isdigit() and len(l.split(',')[0].strip()) == 8
    first = next(i for i, l in enumerate(lines) if is_day(l))
    names = [c.strip() for c in lines[first - 1].split(',')[1:]]
    rows = [l.split(',') for l in lines[first:] if is_day(l)]
    return pd.DataFrame([[float(v) for v in r[1:]] for r in rows], columns=names,
                        index=pd.to_datetime([r[0].strip() for r in rows], format='%Y%m%d'))

def fetch_ff_factors(start_date, end_date):
    # Daily factor returns in decimals, columns:
    #   Mkt-RF  market return minus the risk-free rate
    #   SMB     size (small minus big)          HML  value (high minus low book-to-market)
    #   RMW     profitability (robust - weak)   CMA  investment (conservative - aggressive)
    #   RF      risk-free rate (1-month T-bill) Mom  momentum (winners minus losers)
    print("Fetching Fama-French factors...")
    try:
        if os.path.exists(FF5_FILE) and os.path.exists(MOM_FILE):
            ff5 = read_french_csv(FF5_FILE).loc[start_date:end_date]
            mom = read_french_csv(MOM_FILE).loc[start_date:end_date]
        else:
            import pandas_datareader.data as web
            # F-F Research Data 5 Factors (Mkt-RF, SMB, HML, RMW, CMA, RF)
            ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start_date, end_date)[0]

            # Momentum Factor (Mom)
            mom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench', start_date, end_date)[0]
        
        # Combine
        factors = pd.concat([ff5, mom], axis=1)
        
       
        # The library publishes returns in per cent: convert to decimals
        factors = factors / 100.0

        # pandas_datareader may return a PeriodIndex: convert it so dates match the SPY data
        if isinstance(factors.index, pd.PeriodIndex):
            factors.index = factors.index.to_timestamp()

        return factors.dropna()
    except Exception as e:
        print(f"Error fetching F-F data: {e}")
        return pd.DataFrame()

def analyze_performance(hmm_states, factor_data):
    """
    Calculates performance of strategies in each HMM state.
    """
    
    # hmm_states: regime of each day (Series indexed by date)
    # factor_data: daily factor returns (from fetch_ff_factors)
    # Keep only the days present in both, matched by date
    common_idx = hmm_states.index.intersection(factor_data.index)
    states = hmm_states.loc[common_idx]
    factors = factor_data.loc[common_idx]
    
    
    factors.columns = [c.strip() for c in factors.columns]

    rf = factors['RF']
    mkt_rf = factors['Mkt-RF']
    smb = factors['SMB']
    hml = factors['HML']
    rmw = factors['RMW']
    cma = factors['CMA'] 
    mom = factors['Mom']
    
    # Daily return of each strategy: the market plus a unit exposure to each factor,
    # plus the risk-free rate (the factors are excess or long-short returns)
    #   Fama-French = market + size + value
    #   Carhart     = Fama-French + momentum
    #   Value       = market + value
    #   AQR-style   = market + value + momentum + profitability + investment
    rets_ff = mkt_rf + smb + hml + rf
    
    rets_carhart = mkt_rf + smb + hml + mom + rf
    
    rets_value = mkt_rf + hml + rf

    
    rets_aqr = mkt_rf + hml + mom + rmw + cma + rf
    
    strategies = pd.DataFrame({
        'Fama-French': rets_ff,
        'Carhart': rets_carhart,
        'Value': rets_value,
        'AQR': rets_aqr,
        'Market': mkt_rf + rf, # Baseline
        'RF': rf
    }, index=common_idx)
    
    # Regime of each day next to the returns of the strategies
    strategies['State'] = states
    
    
    results = []
    unique_states = np.sort(strategies['State'].unique())
    
    # For each regime: annualised mean return and Sharpe ratio of each strategy,
    # computed only on the days of that regime
    for state in unique_states:
        state_data = strategies[strategies['State'] == state]
        
        row = {'State': state, 'Count': len(state_data)}
        
        for strat in ['Fama-French', 'Carhart', 'Value', 'AQR', 'Market']:
            rets = state_data[strat]
            avg_ret = rets.mean() * 252 # Annualized (252 trading days a year)
            vol = rets.std() * np.sqrt(252) # Annualized
            sharpe = (avg_ret - (state_data['RF'].mean() * 252)) / vol if vol > 0 else 0
            
            # Annualised mean over annualised standard deviation of daily returns
            sharpe_simple = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
            
            row[f'{strat}_Sharpe'] = sharpe_simple
            row[f'{strat}_Ret'] = avg_ret
        
        results.append(row)
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # In-sample analysis: regimes estimated on 2007-2017 and best strategy in each regime
    # 1. Train HMM and Get States (2007-2017)
    df = fetch_data(start_date="2000-01-01", end_date="2024-01-01")
    df = calculate_features(df).dropna()
    
    train_df = df.loc['2007-01-01':'2017-12-31'].copy()
    X_train = train_df[['Daily_Return', 'Volatility_10d_MSE']].values
    
    hmm = MarketHMM()
    hmm.train(X_train)
    
    # Predict states for the training period
    states = hmm.predict(X_train)
    train_df['HMM_State'] = states
    
    # Check HMM state meanings (Mean Returns)
    state_means = train_df.groupby('HMM_State')['Daily_Return'].mean()
    print("\nState Mean Daily Returns:")
    print(state_means)
    
    # 2. Get Fama-French Data
    ff_data = fetch_ff_factors('2007-01-01', '2017-12-31')
    
    if not ff_data.empty:
        # 3. Benchmark Strategies
        perf = analyze_performance(train_df['HMM_State'], ff_data)
        
        
        print("\nStrategy Performance per Regime (Sharpe Ratio):")
        print(perf.set_index('State')[['Fama-French_Sharpe', 'Carhart_Sharpe', 'Value_Sharpe', 'AQR_Sharpe', 'Market_Sharpe']])
        
        print("\nStrategy Annualized Returns per Regime:")
        print(perf.set_index('State')[['Fama-French_Ret', 'Carhart_Ret', 'Value_Ret', 'AQR_Ret', 'Market_Ret']])
        
        # Identifying Winners
        # (the strategy with the highest Sharpe ratio in each regime, market excluded)
        print("\nBest Performing Strategy per Regime:")
        for _, row in perf.iterrows():
            state = row['State']
            sharpes = {k.replace('_Sharpe', ''): v for k, v in row.items() if '_Sharpe' in k and 'Market' not in k} # Exclude Market baseline if we only want sub-strategies? User said "establish which sub-strategies... Fama, Carhart, Value"
            
            best_strat = max(sharpes, key=sharpes.get)
            print(f"State {state} ({state_means[state]*100:.4f}% Daily Avg): Best = {best_strat} (Sharpe: {sharpes[best_strat]:.2f})")
    
    else:
        print("Failed to retrieve Fama-French data.")
