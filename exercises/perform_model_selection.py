from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    min_range, max_range = -1.2, 2.

    X = np.linspace(min_range, max_range, n_samples)
    noise_addition = np.random.normal(0, noise, size=n_samples)

    y = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)

    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y + noise_addition), 2/3)

    # Back to Numpy format
    X_train = X_train.to_numpy().flatten()
    y_train = y_train.to_numpy().flatten()
    X_test = X_test.to_numpy().flatten()
    y_test = y_test.to_numpy().flatten()

    fig = go.Figure()
    fig.add_traces([go.Scatter(x=X, y=y, mode='markers', name='True f(x)', marker=dict(size=10)),
                    go.Scatter(x=X_train, y=y_train, mode='markers', name='Train Data'),
                    go.Scatter(x=X_test, y=y_test, mode='markers', name='Test Data')])
    fig.update_layout(title=dict(text=f'Data With Noise {noise}'))
    fig.show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    degrees = 10
    train_score, validation_score = np.zeros(degrees + 1), np.zeros(degrees + 1)

    for i in range(degrees + 1):
        estimator = PolynomialFitting(i)
        train_score[i], validation_score[i] = cross_validate(estimator, X_train, y_train, mean_square_error, cv=5)

    fig = go.Figure()
    fig.add_traces([go.Scatter(x=np.arange(degrees + 1), y = train_score, mode='lines', name='Train Score'),
                    go.Scatter(x=np.arange(degrees + 1), y = validation_score, mode='lines', name='Validation Score')])
    fig.update_layout(title=dict(text="Loss of Polynomial Fitting"))
    fig.update_xaxes(title=dict(text="Polynomial Degree"))
    fig.update_yaxes(title=dict(text="Loss"))
    fig.show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    raise NotImplementedError()
