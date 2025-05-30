from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.stats import norm, ranksums, mannwhitneyu
from transformers import PreTrainedModel

from src.utils.UtilitiesPytorch import move_batch_to_device


class StatisticalTest(ABC):
    @abstractmethod
    def __init__(self, data):
        """Initialize the statistical test with the given data."""
        pass

    @abstractmethod
    def evaluate_test(self, **args):
        """Calculate the test statistic."""
        pass

    @abstractmethod
    def print(self):
        """Report the results of the statistical test."""
        pass


class RanksumsTest(StatisticalTest):

    def __init__(self, loss):
        self.loss = loss # Require to have already instantiate the class to make it compatible with loss that are simple function.
        self.pvalue = None

    def evaluate_test(self, model, local_dataloader, remote_dataloader):
        try:
            local_atomic_errors = compute_atomic_errors(model, local_dataloader, self.loss).cpu()
            remote_atomic_errors = compute_atomic_errors(model, remote_dataloader, self.loss).cpu()
        except AttributeError:
            local_atomic_errors = compute_atomic_errors(model, local_dataloader, self.loss)
            remote_atomic_errors = compute_atomic_errors(model, remote_dataloader, self.loss)
        self.pvalue = ranksums(local_atomic_errors, remote_atomic_errors).pvalue
        return self.pvalue

    def print(self):
        pass


class Mannwhitneyu(StatisticalTest):

    def __init__(self, loss):
        self.loss = loss # Require to have already instantiate the class to make it compatible with loss that are simple function.
        self.pvalue = None

    def evaluate_test(self, local_model, remote_model, local_dataloader, alternative="two-sided"):
        try:
            local_atomic_errors = compute_atomic_errors(local_model, local_dataloader, self.loss).cpu()
            remote_atomic_errors = compute_atomic_errors(remote_model, local_dataloader, self.loss).cpu()
        except AttributeError:
            local_atomic_errors = compute_atomic_errors(local_model, local_dataloader, self.loss)
            remote_atomic_errors = compute_atomic_errors(remote_model, local_dataloader, self.loss)
        self.pvalue = mannwhitneyu(local_atomic_errors, remote_atomic_errors, alternative=alternative).pvalue
        return self.pvalue

    def print(self):
        pass


class ProportionTest(StatisticalTest):

    def __init__(self, beta0, delta, loss):
        self.beta0 = beta0
        self.delta = delta
        self.loss = loss # Require to have already instantiate the class to make it compatible with loss that are simple function.
        self.atomic_errors = None
        self.beta_estimator = None
        self.pvalue = None
        self.beta_critique = None

    def evaluate_test(self, q0, remote_model, dataloader):
        if isinstance(remote_model, torch.nn.Module):
            n = len(dataloader.dataset)
        elif hasattr(remote_model, 'predict'):
            n = len(dataloader[1])
        else:
            raise ValueError("Unsupported regressor type.")

        try:
            self.atomic_errors = compute_atomic_errors(remote_model, dataloader, self.loss).cpu()
        except AttributeError:
            self.atomic_errors = compute_atomic_errors(remote_model, dataloader, self.loss)
        if len(self.atomic_errors) != n:
            raise ValueError("The number of atomic errors is not equal to the number of labels.")
        self.beta_estimator = np.sum([1 if e <= q0 else 0 for e in self.atomic_errors]) / n
        self.pvalue = norm.cdf(np.sqrt(n) *
                               (self.beta_estimator - self.beta0 + self.delta) /
                               np.sqrt((self.beta0 - self.delta) * (1 - self.beta0 + self.delta)))
        self.beta_critique = (norm.ppf(0.05) *
                              np.sqrt( (self.beta0 - self.delta) * (1 - self.beta0 + self.delta)) /
                              np.sqrt(n) + self.beta0 - self.delta)

        # plot_arrow_with_atomic_errors(local_atomic_errors1, atomic_errors1, self.beta0,
        #                               pvalue=self.pvalue, main_client=1, name=f"{scenario}_quantiles1")

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


def compute_atomic_errors(net, data_loader, loss):
    # Unified interface to handle both scikit-learn and PyTorch.
    if isinstance(net, torch.nn.Module):
        device = next(net.parameters()).device
        # If the regressor is a PyTorch model
        net.eval()  # Set the model to evaluation mode
        atomic_errors = []
        with torch.no_grad():
            if isinstance(net, PreTrainedModel):
                for batch in data_loader:
                    outputs = net(**move_batch_to_device(batch, device))
                    atomic_errors.append(loss(outputs.logits, move_batch_to_device(batch, device)["labels"]))
            else:
                for x_batch, y_batch in data_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    predictions = net(x_batch)
                    atomic_errors.append(loss(predictions, y_batch))
            atomic_errors = torch.concat(atomic_errors)
    elif hasattr(net, 'predict'):
        # If the regressor is a scikit-learn model
        features, Y = data_loader
        predictions = net.predict(features)
        atomic_errors = [loss(ypred, y) for (y, ypred) in zip(Y, predictions)]
    else:
        raise ValueError("Unsupported regressor type.")

    return atomic_errors