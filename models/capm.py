'''
CAPM Model
'''

#Это программа
import numpy as np
import getFamaFrenchFactors as gff
import statsmodels.api as sm
import pandas as pd


class CAPM:
    def __init__(self, df) -> None:
        """
        Initialize the CAPM model with the risk-free rate and the expected market return.

        Parameters:
        - risk_free_rate: The risk-free rate (e.g., the yield on a 3-month Treasury bill).
        - market_return: The expected return of the market.
        """
        self.factor_names = ['Mkt-RF']
        self.factors = pd.DataFrame(gff.carhart4Factor(frequency='m'))
        self.factors = self.factors.rename(columns={'date_ff_factors': 'Date'})
        self.df = df
        self.beta = None  # Beta coefficient will be calculated during fitting

    def fit(self, ticker, show_info=False) -> None:
        """
        Fit the CAPM model to the given asset returns and market returns.

        Parameters:
        - returns: An array-like object of asset returns.
        - market_returns: An array-like object of market returns corresponding to the asset returns.

        Returns:
        - None
        """
        mon = pd.DataFrame(self.df[ticker]).resample('ME').last()
        mon_rets = mon.pct_change().dropna()
        data = pd.merge(mon_rets, self.factors, on='Date', how='left')
        data = data.set_index('Date')
        data = data.dropna()
        self.beta = (data[ticker] - data['RF']) / self.factors['Mkt-RF']

        Y = data[ticker] - data['RF']  # substract the risk free rate
        X = data[self.factor_names]

        # build model
        self.model = sm.OLS(Y, X)
        result = self.model.fit()

        if show_info:
            print(result.summary())
            avr = self.factors.drop('Date', axis=1).apply(np.mean)
            self.beta = result.params
            exp_returns = self.beta*avr['Mkt-RF']
            e_rets = exp_returns - data['RF'].mean()
            print(f'The expected monthly return for {ticker} is:', e_rets)
            print(f'The expected anual return for {ticker} is:', ((
                1 + e_rets) ** 12) - 1)

        return result

    def predict(self, X) -> np.ndarray:
        """
        Predict the expected returns of an asset using the fitted CAPM model.

        Parameters:
        - returns: An array-like object of asset returns.

        Returns:
        - An array-like object of predicted expected returns.
        """
        # Calculate the expected returns using the CAPM formula

        # expected_returns = self.risk_free_rate + self.beta * \
        #     (self.market_return - self.risk_free_rate)
        # return expected_returns
        ...
