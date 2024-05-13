import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import ARDRegression


class RFEARDRegression:
    """
    A class that combines Recursive Feature Elimination (RFE) with Adaptive Regression (ARD) for feature selection and model fitting.

    Attributes:
        n_features_to_select (int): The number of features to select using RFE.
        step (int): The step size for RFE.
        estimator (SVR): The estimator used for RFE, set to a linear SVR.
        selector (RFE): The RFE object for feature selection.
        ard (ARDRegression): The ARDRegression model for final fitting.
    """

    def __init__(self, n_features_to_select=5, step=1):
        """
        Initializes the RFEARDRegression class.

        Parameters:
            n_features_to_select (int): Number of features to select.
            step (int): Step size for RFE.
        """
        self.n_features_to_select = n_features_to_select
        self.step = step
        # Linear Support Vector Regression for RFE
        self.estimator = SVR(kernel="linear")
        self.selector = RFE(
            self.estimator, n_features_to_select=self.n_features_to_select, step=self.step)  # RFE for feature selection
        self.ard = ARDRegression()  # Adaptive Regression model for final fitting

    def fit(self, X_train, y_train):
        """
        Fits the model to training data.

        Parameters:
            X_train (numpy.ndarray): Training data.
            y_train (numpy.ndarray): Training labels.

        Returns:
            ARDRegression: The fitted ARDRegression model.
        """
        # Apply RFE to select features
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        # Fit the ARDRegression model on the selected features
        self.ard.fit(X_train_selected, y_train)
        return self.ard

    def predict(self, X_test):
        """
        Predicts labels for test data.

        Parameters:
            X_test (numpy.ndarray): Test data.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        # Transform the test data using the same feature selection process
        X_selected = self.selector.transform(X_test)
        # Use the fitted ARDRegression model to predict labels
        return self.ard.predict(X_selected)
