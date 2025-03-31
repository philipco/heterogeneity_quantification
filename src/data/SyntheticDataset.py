# Custom PyTorch Dataset for generating synthetic data on the fly
import torch
from numpy.random import multivariate_normal
from scipy.stats import ortho_group
from torch.utils.data import IterableDataset


class SyntheticLSRDataset(IterableDataset):

    def __init__(self, true_theta, batch_size, noise_std=2):
        """
        Parameters:
        - true_theta: (d,) torch.Tensor, the true parameter vector
        - batch_size: int, number of samples per batch
        - noise_std: float, standard deviation of Gaussian noise
        """

        self.dim = true_theta.shape[0]
        self.true_theta = true_theta

        self.eigenvalues = torch.FloatTensor([1 / (i**2) for i in range(1, self.dim + 1)[::-1]])
        self.covariance = torch.diag(self.eigenvalues)
        self.ortho_matrix = torch.FloatTensor(ortho_group.rvs(dim=self.dim, random_state=5))
        self.covariance = self.ortho_matrix @ self.covariance @ self.ortho_matrix.T

        self.batch_size = batch_size
        self.noise_std = noise_std

    def __iter__(self):
        while True:
            X = torch.FloatTensor(multivariate_normal(torch.zeros(self.dim), self.covariance, size=self.batch_size))
            # X = torch.randn(self.batch_size, self.dim, self.covariance)  # Features from N(0, I)
            noise = torch.normal(mean=0, std=self.noise_std, size=(self.batch_size, ))
            y = X @ self.true_theta + noise  # Generate targets
            yield X, y.reshape(self.batch_size, 1) # Yield a batch instead of returning

    def __len__(self):
        return None #float('inf')  # Arbitrary value, as it's an infinite generator