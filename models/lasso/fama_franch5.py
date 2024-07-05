'''
Fama-French 5-Factor Model

Factors loaded from: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html#Research
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import getFamaFrenchFactors as gff
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from ..fama_franch5 import FamaFrench5Factor


class FamaFrench5FactorLasso(FamaFrench5Factor):
    def __init__(self, df) -> None:
        super().__init__(df=df)

    def fit_lasso(self, ticker, show_info=False):
        data = self._concat_stock_with_factors(ticker)
        Y = data[ticker] - data['RF']  # subtract the risk-free rate
        X = data[self.factor_names]

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform Lasso CV regression
        lasso_cv = LassoCV(cv=5, random_state=42)
        lasso_cv.fit(X_scaled, Y)

        if show_info:
            print("Lasso CV Regression Results:")
            print(lasso_cv.coef_)
            print("Best alpha:", lasso_cv.alpha_)

            # Predicting new values
            predictions = lasso_cv.predict(scaler.transform(X))

            # Calculating expected returns
            exp_returns = np.dot(
                predictions, self.factors[self.factor_names]) + data['RF'].mean()
            print(
                f'The expected monthly return for {ticker} is:', exp_returns[-1])
            print(f'The expected annual return for {ticker} is:', ((
                1 + exp_returns[-1]) ** 12) - 1)

        return lasso_cv
