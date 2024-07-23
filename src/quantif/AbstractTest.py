"""Created by Constantin Philippenko, 16th July 2024."""

from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.stats import norm, chi2

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

    def __init__(self, beta0, loss):
        self.beta0 = beta0
        self.loss = loss # Require to have already instantiate the class to make it compatible with loss that are simple function.
        self.atomic_errors = None
        self.beta_estimator = None
        self.pvalue = None
        self.beta_critique = None

    def evaluate_test(self, q0, remote_model, features, labels):
        n = len(labels)
        self.atomic_errors = compute_atomic_errors(remote_model, features, labels, self.loss)
        if len(self.atomic_errors) != len(labels):
            raise ValueError("The number of atomic errors is not equal to the number of labels.")
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


def compute_atomic_errors(reg, features, Y, loss):
    # Unified interface to handle both scikit-learn and PyTorch.
    if isinstance(reg, torch.nn.Module):
        # If the regressor is a PyTorch model
        reg.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = reg(features)
            atomic_errors = loss(predictions, Y)
    elif hasattr(reg, 'predict'):
        # If the regressor is a scikit-learn model
        predictions = reg.predict(features)
        atomic_errors = [loss(ypred, y) for (y, ypred) in zip(Y, predictions)]
    else:
        raise ValueError("Unsupported regressor type.")

    return atomic_errors