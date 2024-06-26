'''
Fama-French 3-Factor Model

Factors loaded from: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#Research
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import getFamaFrenchFactors as gff


class FamaFrench3Factor:
    def __init__(self, df) -> None:
        self.factors = pd.DataFrame(gff.famaFrench3Factor(frequency='m'))
        self.factors = self.factors.rename(columns={'date_ff_factors': 'Date'})
        self.df = df

    def fit(self, ticker, show_info=False):
        data = self._concat_stock_with_factors(ticker)
        factors = ['Mkt-RF', 'SMB', 'HML']
        Y = data[ticker] - data['RF']  # substract the risk free rate
        X = data[factors]

        # build model
        self.model = sm.OLS(Y, sm.tools.add_constant(X))
        result = self.model.fit()

        if show_info:
            print(result.summary())
            avr = self.factors.drop('Date', axis=1).apply(np.mean)
            Int, Mkt, SMB, HML = result.params
            exp_returns = Mkt*avr['Mkt-RF'] + SMB*avr['SMB'] + HML*avr['HML']
            e_rets = exp_returns - data['RF'].mean()
            print(f'The expected monthly return for {ticker} is:', e_rets)
            print(f'The expected anual return for {ticker} is:', ((
                1 + e_rets) ** 12) - 1)

        return result

    def predict(self, X) -> np.array:
        # Ensure X has the same columns as the factors used in the model
        if not set(X.columns).issubset(set(self.factors.columns)):
            raise ValueError(
                "Input DataFrame must contain the same columns as the factors used in the model.")

        # Add a constant term to the input DataFrame
        X = sm.tools.add_constant(X)

        # Use the fitted model to predict the returns
        predictions = self.model.predict(X)

        return predictions

    def _concat_stock_with_factors(self, ticker):
        mon = pd.DataFrame(self.df[ticker]).resample('ME').last()
        mon_rets = mon.pct_change().dropna()
        data = pd.merge(mon_rets, self.factors, on='Date', how='left')
        data = data.set_index('Date')
        data = data.dropna()
        return data

    def get_params(self):
        return self.model.fit().params

    def factors_plot(self, ticker) -> None:
        # factors = ['Mkt-RF', 'SMB', 'HML']

        data = self._concat_stock_with_factors(ticker)

        # Plot Fama-French factors
        plt.figure(figsize=(20, 10))
        fig3, axs = plt.subplots(1, 3)
        axs[0].plot(data['Mkt-RF'].rolling(3).mean(), linewidth=2.5)
        axs[0].plot(data[ticker])
        axs[0].set_title('Stock compared to market returns')
        axs[1].plot(data['SMB'].rolling(3).mean(), linewidth=2.5)
        axs[1].plot(data[ticker])
        axs[1].set_title('Stock compared to small company returns')
        axs[2].plot(data['HML'].rolling(3).mean(), linewidth=2.5)
        axs[2].plot(data[ticker])
        axs[2].set_title('Stock compared to value stocks index')
        fig3.suptitle('Factors plot', fontsize=18)

        # Calculate correlations
        cor = data.corr()

        # Use.iloc[] to access values by their integer location
        print(f'{ticker} correlation with market index:',
              cor['Mkt-RF'].iloc[0])
        print(f'{ticker} correlation with small-company portfolio index:',
              cor['SMB'].iloc[0])
        print(f'{ticker} correlation with value stocks index:',
              cor['HML'].iloc[0])

    # TESTS: f-test, t-test

    def f_test(self) -> None:
        pass
