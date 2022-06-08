from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    # Splitting and declaring arrays to receive the scoring
    X_split = np.array_split(X, cv)
    y_split = np.array_split(y, cv)
    val_score, train_score = np.zeros(cv), np.zeros(cv)

    for i in range(cv):
        # Making train and validation separate
        X_validate = X_split[i]
        y_validate = y_split[i]
        X_train = np.concatenate(X_split[:i] + X_split[i+1:])
        y_train = np.concatenate(y_split[:i] + y_split[i+1:])

        estimator.fit(X_train, y_train)

        train_pred = estimator.predict(X_train)
        val_pred = estimator.predict(X_validate)

        train_score[i] = scoring(y_train, train_pred)
        val_score[i] = scoring(y_validate, val_pred)

    return np.average(train_score), np.average(val_score)
