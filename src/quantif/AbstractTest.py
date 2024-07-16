"""Created by Constantin Philippenko, 16th July 2024."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm

class StatisticalTest(ABC):
    @abstractmethod
    def __init__(self, data):
        """Initialize the statistical test with the given data."""
        pass

    @abstractmethod
    def evaluate_test(self):
        """Calculate the test statistic."""
        pass

    @abstractmethod
    def print(self):
        """Report the results of the statistical test."""
        pass


class ProportionTest(StatisticalTest):

    def __init__(self, beta0):
        self.beta0 = beta0
        self.atomic_errors = None
        self.beta_estimator = None
        self.pvalue = None
        self.beta_critique = None

    def evaluate_test(self, q0, remote_model, features, labels):
        n = len(labels)
        self.atomic_errors = compute_atomic_errors(remote_model, features, labels)

        self.beta_estimator = np.sum([e <= q0 for e in self.atomic_errors]) / n
        self.pvalue = norm.cdf(np.sqrt(n) * (self.beta_estimator - self.beta0) / np.sqrt(self.beta0 * (1 - self.beta0)))
        self.beta_critique = norm.ppf(0.05) * np.sqrt(self.beta0 * (1 - self.beta0)) / np.sqrt(n) + self.beta0

    def print(self):
        print("=   Proportion test   =")
        if self.pvalue is None:
            return Exception("The test has not been evaluated.")
        if self.pvalue < 0.05:
            print("=> H0 is rejected.")
        else:
            print("=> H0 can not be rejected.")

        print(f"\tEstimation de beta: {self.beta_estimator}")
        print(f"\tP-value: {self.pvalue}")
        print(f"\tBeta critique: {self.beta_critique}")


def MSE(y, ypred):
    return (y - ypred)**2 / 2


def sigmoid_loss(y, y_pred):
    return np.log(1 + np.exp(-y * y_pred))

def compute_atomic_errors(reg, features, Y, logistic=False):
    prediction = reg.predict(features)
    if not logistic:
        atomic_errors = [MSE(y, ypred) for (y, ypred) in zip(Y, prediction)]
    else:
        atomic_errors = [sigmoid_loss(y, ypred) for (y, ypred) in zip(Y, prediction)]
    return atomic_errors