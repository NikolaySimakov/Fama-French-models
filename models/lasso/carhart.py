'''
Carhart four-factor model is an extra factor addition in the Fama–French three-factor model

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
from sklearn.pipeline import Pipeline


class CarhartLasso:
    def __init__(self, df) -> None:
        self.factor_names = ['Mkt-RF', 'SMB', 'HML', 'MOM']
        self.factors = pd.DataFrame(gff.carhart4Factor(frequency='m'))
        self.factors = self.factors.rename(columns={'date_ff_factors': 'Date'})
        self.df = df

    def fit(self, ticker, show_info=False):
        data = self._concat_stock_with_factors(ticker)
        Y = data[ticker] - data['RF']  # subtract the risk-free rate
        X = data[self.factor_names]

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Define the pipeline
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lasso_cv', LassoCV(cv=5))  # Use 5-fold cross-validation
        ])

        # Fit the model
        pipe.fit(X_scaled, Y)

        if show_info:
            print("Best alpha:", pipe.named_steps['lasso_cv'].alpha_)
            print("Coefficients:", pipe.named_steps['lasso_cv'].coef_)

            # Predictions
            predictions = pipe.predict(scaler.transform(X))

            # Expected returns calculation remains the same
            avr = self.factors.drop('Date', axis=1).apply(np.mean)
            exp_returns = pipe.named_steps['lasso_cv'].coef_[0]*avr['Mkt-RF'] + \
                pipe.named_steps['lasso_cv'].coef_[1]*avr['SMB'] + \
                pipe.named_steps['lasso_cv'].coef_[2]*avr['HML'] + \
                pipe.named_steps['lasso_cv'].coef_[3]*avr['MOM']
            e_rets = exp_returns - data['RF'].mean()
            print(f'The expected monthly return for {ticker} is:', e_rets)
            print(f'The expected annual return for {ticker} is:', ((
                1 + e_rets) ** 12) - 1)

        return pipe

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
        data = self._concat_stock_with_factors(ticker)

        # Plot Fama-French factors
        plt.figure(figsize=(20, 10))
        fig3, axs = plt.subplots(1, 4)
        axs[0].plot(data['Mkt-RF'].rolling(3).mean(), linewidth=2.5)
        axs[0].plot(data[ticker])
        axs[0].set_title('Stock compared to market returns')
        axs[1].plot(data['SMB'].rolling(3).mean(), linewidth=2.5)
        axs[1].plot(data[ticker])
        axs[1].set_title('Stock compared to small company returns')
        axs[2].plot(data['HML'].rolling(3).mean(), linewidth=2.5)
        axs[2].plot(data[ticker])
        axs[2].set_title('Stock compared to value stocks index')
        axs[3].plot(data['MOM'].rolling(3).mean(), linewidth=2.5)
        axs[3].plot(data[ticker])
        axs[3].set_title('Stock compared to momentum')
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
        print(f'{ticker} correlation with momentum:',
              cor['MOM'].iloc[0])
