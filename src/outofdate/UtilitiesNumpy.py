"""Created by Constantin Philippenko, 30th September 2022."""
import math

import numpy as np
from batchup import data_source
from sklearn.decomposition import IncrementalPCA

def fit_PCA(features: np.array, ipca_data: IncrementalPCA, scaler, batch_size: int) -> IncrementalPCA:
    ds = data_source.ArrayDataSource([features])
    for x in ds.batch_iterator(batch_size=batch_size, shuffle=False):
        x = x[0]
        if len(x.shape) > 2:
            x = np.flatten(x, start_dim=1)
        if scaler is not None:
            x = scaler.transform(x)

        # If there is less features in the dataset than the wished numbers of PCA components, we return None.
        if x.shape[1] <= ipca_data.n_components:
            return None

        # To fit the PCA we must have a number of elements bigger than the PCA dimension, those we must drop the last
        # batch if it doesn't contain enough elements.
        if ipca_data is not None and x.shape[0] >= ipca_data.n_components:
            ipca_data.partial_fit(x)
    return ipca_data


def compute_PCA(features: np.array, ipca_data: IncrementalPCA, scaler, batch_size: int) -> np.ndarray:
    X, Y = [], []
    ds = data_source.ArrayDataSource([features])
    for x in ds.batch_iterator(batch_size=batch_size, shuffle=False):
        x = x[0]
        if len(x.shape) > 2:
            x = np.flatten(x, start_dim=1)
        if scaler is not None:
            x = scaler.transform(x)
        if ipca_data is not None:
            X.append(ipca_data.transform(x))
        else:
            X.append(x) # TODO reshape : .reshape(-1, x.shape[0] * x.shape[1]))
    return np.concatenate(X)


def remove_diagonal(distribution: np.array, symmetric_matrix) -> np.array:

    # If the matrix is symmetric, we also need to remove symmetric elements.
    if symmetric_matrix:
        r, c = np.triu_indices(len(distribution), 1)
        return distribution[r, c]

    distrib_without_diag = distribution.flatten()
    return np.delete(distrib_without_diag, range(0, len(distrib_without_diag), len(distribution) + 1), 0)


def create_matrix_with_zeros_diagonal_from_array(distribution: np.array) -> np.array:
    d = list(np.concatenate(distribution))
    size = (1 + math.sqrt(1 + 4 * len(d))) / 2
    assert size.is_integer(), "The size is not an integer."
    size = int(size)
    for i in range(size):
        d.insert(i * size + i, 0)
    return np.array(d).reshape((size, size))
