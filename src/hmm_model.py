# hmm_model.py
# Gaussian Hidden Markov Model of market regimes.
#
# The market is assumed to move between a few hidden states (regimes). In each state the
# observations (daily return, volatility) follow a bivariate normal distribution with their
# own mean and covariance; a transition matrix gives the probability of moving from one
# state to another the next day. The parameters are estimated with the Baum-Welch (EM)
# algorithm, and the most likely sequence of states with the Viterbi algorithm.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from data_loader import fetch_data, calculate_features


class MarketHMM:
    # Wrapper around hmmlearn's GaussianHMM.
    #   n_components     number of hidden states (regimes)
    #   covariance_type  "full": return and volatility can be correlated within a state
    #   n_iter           maximum number of EM iterations
    #   random_state     seed of the starting point of EM, for reproducible results
    def __init__(self, n_components=2, covariance_type="full", n_iter=75, random_state=42):
        self.model = GaussianHMM(n_components=n_components,
                            covariance_type=covariance_type,
                            n_iter=n_iter,
                            random_state=random_state)

    def train(self, data):
        # Estimates means, covariances, transition matrix and initial probabilities.
        # data: array with one row per day and two columns (return, volatility)
        print(f"Training HMM with {self.model.n_components} components...")
        self.model.fit(data)
        print("Model converged:", self.model.monitor_.converged)

    def predict(self, data):
        # Most likely state of each day (Viterbi). State numbers are arbitrary: their
        # meaning (bull, bear...) is read from the mean return of each state.
        return self.model.predict(data)

    def get_state_stats(self):
        # Fitted parameters: mean and covariance of each state, and transition matrix
        print(self.model.transmat_)
        return self.model.means_, self.model.covars_, self.model.transmat_

if __name__ == "__main__":
    # In-sample analysis: fit the HMM on 2007-2017 and look at the regimes it finds
    # 1. Load Data
    df = fetch_data(start_date="2000-01-01", end_date="2024-01-01")
    df = calculate_features(df).dropna()

    # 2. Filter for Training Period (2007-2017)
    train_df = df.loc['2007-01-01':'2017-12-31'].copy()

    if train_df.empty:
        print("Error: No training data found for 2007-2017.")
    else:
        # Observations: one row per day, columns = daily return and 10-day volatility
        X_train = train_df[['Daily_Return', 'Volatility_10d_MSE']].values

        # 3. Train Model
        hmm_model = MarketHMM()
        hmm_model.train(X_train)

        # 4. Concatenazione (Richiesta)
        # Regime of each day, added as a column next to the data
        hidden_states = hmm_model.predict(X_train)
        states_series = pd.Series(hidden_states, index=train_df.index, name='Regime')
        train_results = pd.concat([train_df, states_series], axis=1)


        # Distribution (kernel density) of returns and volatility in each regime
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))


        for regime in sorted(train_results['Regime'].unique()):
            subset = train_results[train_results['Regime'] == regime]
            sns.kdeplot(subset['Daily_Return'], ax=axes[0], label=f'Regime {regime}', fill=True, bw_adjust=1.5)

        axes[0].set_title('PDF of daily returns per regime', fontsize=14)
        axes[0].set_xlabel('Daily Return')
        axes[0].set_xlim(-0.06, 0.06)
        axes[0].set_ylabel('Density')
        axes[0].legend()



        for regime in sorted(train_results['Regime'].unique()):
            subset = train_results[train_results['Regime'] == regime]
            sns.kdeplot(subset['Volatility_10d_MSE'], ax=axes[1], label=f'Regime {regime}', fill=True, bw_adjust=1.5)

        axes[1].set_title('Volatility PDF per Regime', fontsize=14)
        axes[1].set_xlabel('Volatility (MSE)')
        axes[1].set_ylabel('Density')
        axes[1].legend()

        plt.tight_layout()
        plt.show()



        # Mean return and volatility of each regime: the regime with the lowest mean
        # return and highest volatility is the bear market
        print(train_results.groupby('Regime')[['Daily_Return', 'Volatility_10d_MSE']].mean())


