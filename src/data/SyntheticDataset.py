"""
This module defines a PyTorch `IterableDataset` class to generate synthetic
linear regression data on-the-fly, designed for simulating an online training
scenario matching the theoretical framework analysed in our paper.

The data points are generated from a multivariate normal distribution with a
prescribed covariance structure. The target labels are computed using a
fixed ground-truth parameter vector `true_theta`. Additionally, the Lipschitz
smoothness constant and the strong convexity parameter (μ) of the least-squares
loss are estimated using the empirical covariance of the features.
"""

import torch
from numpy.random import multivariate_normal
from scipy.stats import ortho_group
from torch.utils.data import IterableDataset


class SyntheticLSRDataset(IterableDataset):
    """
    PyTorch IterableDataset for generating synthetic linear regression data on-the-fly.

    This dataset yields infinite batches of synthetic data drawn from a multivariate
    Gaussian distribution with a specified covariance structure. Targets are computed
    using a fixed parameter vector `true_theta`. This is suitable for experiments in
    online learning settings.

    Attributes:
        true_theta (torch.Tensor): Ground truth parameter vector (shape: d,).
        variation (torch.Tensor): The model's variation compared to the true model of the cluster.
        batch_size (int): Number of samples per batch.
        noise_std (float): Standard deviation of optional additive Gaussian noise.
        covariance (torch.Tensor): Covariance matrix of the input distribution.
        L (float): Estimated Lipschitz constant of the least-squares loss.
        mu (float): Estimated strong convexity constant of the least-squares loss.
    """

    def __init__(self, true_theta, variation, batch_size: int, noise_std: int = 0.1):
        """
        Initialize the synthetic dataset.

        Args:
            true_theta (torch.Tensor): True regression parameter of shape (d,).
            variation (torch.Tensor): The model's variation compared to the true model of the cluster.
            batch_size (int): Number of samples per yielded batch.
            noise_std (float): Standard deviation for Gaussian noise added to labels.
        """
        self.dim = true_theta.shape[0]
        self.true_theta = true_theta
        self.variation = variation

        # Define descending eigenvalues to build a well-conditioned covariance matrix
        self.eigenvalues = torch.DoubleTensor([1 for i in range(1, self.dim + 1)[::-1]])

        # Construct covariance matrix with orthogonal eigenbasis
        self.covariance = torch.diag(self.eigenvalues)
        self.ortho_matrix = torch.DoubleTensor(ortho_group.rvs(dim=self.dim))
        self.covariance = self.ortho_matrix @ self.covariance @ self.ortho_matrix.T

        self.batch_size = batch_size
        self.noise_std = noise_std

        # Compute smoothness (Lipschitz constant) and strong convexity (mu)
        self.L, self.mu = self.compute_lips_mu()

    def __iter__(self):
        """
        Yield an infinite stream of synthetic (X, y) pairs.

        X is drawn from N(0, Σ) where Σ is the constructed covariance matrix.
        y = X @ true_theta + noise

        Returns:
            Generator[Tuple[torch.Tensor, torch.Tensor]]: Batches of input/output pairs.
        """
        while True:
            X = torch.DoubleTensor(
                multivariate_normal(torch.zeros(self.dim), self.covariance, size=self.batch_size)
            )

            # Optional additive Gaussian noise (commented out for deterministic targets)
            noise = torch.normal(mean=0, std=self.noise_std, size=(self.batch_size,))
            y = X @ self.true_theta  # + noise
            yield X, y.reshape(self.batch_size, 1)

    def __len__(self):
        """
        Return None as the dataset is infinite.

        This makes it compatible with PyTorch's IterableDataset interface.
        """
        return None

    def compute_lips_mu(self):
        """
        Empirically estimate Lipschitz (L) and strong convexity (μ) constants.

        These are derived from the empirical covariance matrix computed over
        multiple generated batches. Useful for theoretical guarantees in convergence
        analysis.

        Returns:
            Tuple[float, float]: Estimated Lipschitz constant and strong convexity constant.
        """
        cov = torch.zeros((self.dim, self.dim))
        nb_samples = 10**4

        # Estimate covariance matrix from nb_samples synthetic batches
        for k in range(nb_samples):
            X = torch.DoubleTensor(
                multivariate_normal(torch.zeros(self.dim), self.covariance, size=self.batch_size)
            )
            cov += X.T.mm(X) / self.batch_size

        cov /= nb_samples

        # Lipschitz constant = 2 * largest eigenvalue (i.e., spectral norm)
        lips = 2 * torch.linalg.svd(cov).S[0].item()
        print("Lipschitz constant:", lips)

        # Strong convexity = 2 * smallest eigenvalue
        mu = 2 * torch.linalg.svd(cov).S[-1].item()
        print("Strong convexity constant:", mu)

        return lips, mu
