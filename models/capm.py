'''
CAPM Model
'''


import numpy as np


class CAPM:
    def __init__(self, risk_free_rate: np.array, market_return: np.array) -> None:
        """
        Initialize the CAPM model with the risk-free rate and the expected market return.

        Parameters:
        - risk_free_rate: The risk-free rate (e.g., the yield on a 3-month Treasury bill).
        - market_return: The expected return of the market.
        """
        self.risk_free_rate = risk_free_rate
        self.market_return = market_return
        self.beta = None  # Beta coefficient will be calculated during fitting

    def fit(self, returns: np.ndarray, market_returns: np.ndarray) -> None:
        """
        Fit the CAPM model to the given asset returns and market returns.

        Parameters:
        - returns: An array-like object of asset returns.
        - market_returns: An array-like object of market returns corresponding to the asset returns.

        Returns:
        - None
        """
        # Calculate the covariance between the asset returns and market returns
        cov_market = np.cov(market_returns, returns)[0, 1]
        # Calculate the variance of the market returns
        var_market = np.var(market_returns)
        # Calculate the beta coefficient
        self.beta = cov_market / var_market

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """
        Predict the expected returns of an asset using the fitted CAPM model.

        Parameters:
        - returns: An array-like object of asset returns.

        Returns:
        - An array-like object of predicted expected returns.
        """
        # Calculate the expected returns using the CAPM formula
        expected_returns = self.risk_free_rate + self.beta * \
            (self.market_return - self.risk_free_rate)
        return expected_returns
