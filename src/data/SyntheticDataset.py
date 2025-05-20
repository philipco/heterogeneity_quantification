# Custom PyTorch Dataset for generating synthetic data on the fly
import torch
from numpy.random import multivariate_normal
from scipy.stats import ortho_group
from torch.utils.data import IterableDataset


class SyntheticLSRDataset(IterableDataset):

    def __init__(self, true_theta, variation, batch_size, noise_std=0.1):
        """
        Parameters:
        - true_theta: (d,) torch.Tensor, the true parameter vector
        - batch_size: int, number of samples per batch
        - noise_std: float, standard deviation of Gaussian noise
        """

        self.dim = true_theta.shape[0]
        self.true_theta = true_theta
        self.variation = variation

        self.eigenvalues = torch.DoubleTensor([1 for i in range(1, self.dim + 1)[::-1]])
        self.covariance = torch.diag(self.eigenvalues)
        self.ortho_matrix = torch.DoubleTensor(ortho_group.rvs(dim=self.dim))
        self.covariance = self.ortho_matrix @ self.covariance @ self.ortho_matrix.T

        self.batch_size = batch_size
        self.noise_std = noise_std

        self.L, self.mu = self.compute_lips_mu()

    def __iter__(self):
        while True:
            X = torch.DoubleTensor(multivariate_normal(torch.zeros(self.dim), self.covariance, size=self.batch_size))

            noise = torch.normal(mean=0, std=self.noise_std, size=(self.batch_size, ))
            y = X @ self.true_theta #+ noise  # Generate targets
            yield X, y.reshape(self.batch_size, 1) # Yield a batch instead of returning

    def __len__(self):
        return None #float('inf')  # Arbitrary value, as it's an infinite generator

    def compute_lips_mu(self):
        cov = torch.zeros((self.dim, self.dim))
        nb_samples = 10**4
        for k in range(nb_samples):
            X = torch.DoubleTensor(multivariate_normal(torch.zeros(self.dim), self.covariance, size=self.batch_size))
            cov += X.T.mm(X) / self.batch_size
        lips = 2 * torch.linalg.svd(cov / nb_samples).S[0].item()
        # lips = 2 * torch.norm(cov / nb_samples, p=2).item()
        print("Lipshitz constant: ", lips)
        mu = 2 * torch.linalg.svd(cov / nb_samples).S[-1].item()
        print("Mu constant:", mu)
        return lips, mu

class StreamingGaussianDataset(IterableDataset):
    def __init__(self, means, dim=2, batch_size=32, num_classes=2):
        self.dim = dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        mean_shift = 2
        self.means = torch.stack([means[i] * mean_shift * torch.ones(dim) for i in range(num_classes)])
        self.cov = torch.eye(dim)  # Identity covariance for simplicity

    def __iter__(self):
        while True:
            while True:
                labels = torch.randint(0, self.num_classes, (self.batch_size,))
                means = self.means[labels]
                samples = torch.normal(means, torch.ones(self.batch_size, self.num_features))
                yield samples, labels