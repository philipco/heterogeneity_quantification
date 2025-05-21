"""
Federated Learning Client Dataset Split Strategies

This module provides various strategies to split datasets across clients
for Federated Learning (FL) experiments. It supports IID, Dirichlet-based
non-IID, and partition-based non-IID splits. Each split produces a list of
data and label tensors corresponding to the dataset allocated to each client.

The module also includes a t-SNE visualization function to explore the
embedding structure of client datasets.
"""

import warnings
from typing import List

import torch
import numpy as np
from sklearn.manifold import TSNE


def iid_split(data: np.ndarray, labels: np.ndarray, nb_clients: int) -> [List[np.ndarray], List[np.ndarray]]:
    """
    Split the dataset IID across clients.

    Each client receives a randomly shuffled, approximately equal share of the data.

    Args:
        data (np.ndarray): Features.
        labels (np.ndarray): Corresponding labels.
        nb_clients (int): Number of clients.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Features and labels for each client.
    """
    nb_points = data.shape[0]

    # Randomly assign proportions to clients then normalize
    proportion_sampling = [np.random.uniform(0, 1) for _ in range(nb_clients)]
    proportion_sampling /= np.sum(proportion_sampling)

    # Determine number of samples per client
    nb_points_by_non_iid_clients = [int(nb_points * p) for p in proportion_sampling]

    X, Y = [], []
    indices = np.arange(nb_points)
    np.random.shuffle(indices)

    # Calculate cumulative split indices
    idx_split = [np.sum(nb_points_by_non_iid_clients[:i]) for i in range(1, nb_clients)]
    split_indices = np.array_split(indices, idx_split)

    # Assign data to each client
    for i in range(nb_clients):
        X.append(data[split_indices[i]])
        Y.append(labels[split_indices[i]])
    return X, Y


def create_non_iid_split(features: List[np.ndarray], labels: List[np.ndarray], nb_clients: int,
                         split_type: str, dataset_name: str) -> [List[np.ndarray], List[np.ndarray]]:
    """
    Factory function to create different types of data splits.

    Args:
        features (List[np.ndarray]): Feature matrix.
        labels (List[np.ndarray]): Corresponding labels.
        nb_clients (int): Number of clients.
        split_type (str): Type of split ("iid", "dirichlet", "partition").
        dataset_name (str): Name of the dataset (affects Dirichlet coefficient).

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Features and labels for each client.
    """
    np.random.seed(2024)

    if split_type == "iid":
        print("IID split.")
        return iid_split(features, labels, nb_clients)

    if split_type == "dirichlet":
        print("Dirichlet split")
        alpha = 0.1 if dataset_name == "mnist" else 1
        return dirichlet_split(features, labels, nb_clients, alpha)

    if split_type == "partition":
        print("Partition split.")
        return sort_and_partition_split(features, labels, nb_clients)


def sort_and_partition_split(features: np.ndarray, labels: np.ndarray, nb_clients: int) \
        -> [List[np.ndarray], List[np.ndarray]]:
    """
    Non-IID split by sorting data by label and partitioning it.

    Each label class is split into multiple chunks, which are then
    cyclically assigned to different clients.

    Args:
        features (np.ndarray): Feature matrix.
        labels (np.ndarray): Corresponding labels.
        nb_clients (int): Number of clients.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Features and labels for each client.
    """
    unique_labels = np.unique(labels)
    nb_of_split_for_one_label = int(np.ceil(2 * nb_clients / len(unique_labels)))

    # Group features by label
    sorted_features, sorted_labels = [], []
    for label in unique_labels:
        sorted_features.append(features[labels == label])
        sorted_labels.append(labels[labels == label])

    X, Y = [[] for _ in range(nb_clients)], [[] for _ in range(nb_clients)]
    counter_client = 0

    for i in range(len(unique_labels)):
        size = len(sorted_labels[i])

        # Split current label's data into chunks
        split_points = [j * size // nb_of_split_for_one_label for j in range(1, nb_of_split_for_one_label)]
        features_split = np.split(sorted_features[i], split_points)
        labels_split = np.split(sorted_labels[i], split_points)

        # Distribute chunks round-robin to clients
        for j in range(nb_of_split_for_one_label):
            X[counter_client % nb_clients].append(features_split[j])
            Y[counter_client % nb_clients].append(labels_split[j])
            counter_client += 1

    # Concatenate all chunks per client
    for idx_client in range(nb_clients):
        X[idx_client] = torch.concat(X[idx_client])
        Y[idx_client] = torch.concat(Y[idx_client])

    return X, Y


def dirichlet_split(data: np.ndarray, labels: np.ndarray, nb_clients: int, dirichlet_coef: float = 1.0) \
        -> [List[np.ndarray], List[np.ndarray]]:
    """
    Non-IID split using Dirichlet distribution over labels.

    Simulates client-specific data preferences where label distributions are sampled
    from a Dirichlet distribution.

    Args:
        data (np.ndarray): Feature matrix.
        labels (np.ndarray): Corresponding labels.
        nb_clients (int): Number of clients.
        dirichlet_coef (float): Dirichlet distribution coefficient (smaller => more heterogeneous).

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Features and labels for each client.
    """
    nb_labels = len(np.unique(labels))  # Total number of unique labels

    X, Y = [[] for _ in range(nb_clients)], [[] for _ in range(nb_clients)]

    for idx_label in range(nb_labels):
        # Sample label proportions for each client
        proportions = np.random.dirichlet(np.repeat(dirichlet_coef, nb_clients))
        assert round(proportions.sum()) == 1, "The sum of proportions is not equal to 1."

        # Get all samples for the current label
        label_data = data[labels == idx_label]
        label_targets = labels[labels == idx_label]
        N = len(label_targets)

        # Compute split indices and assign to clients
        split_indices = [np.sum([int(proportions[k] * N) for k in range(j)]) for j in range(1, nb_clients)]
        features_split = np.split(label_data, split_indices)
        labels_split = np.split(label_targets, split_indices)

        for j in range(nb_clients):
            X[j].append(features_split[j])
            Y[j].append(labels_split[j])

    # Final concatenation for each client
    for idx_client in range(nb_clients):
        X[idx_client] = torch.concat(X[idx_client])
        Y[idx_client] = torch.concat(Y[idx_client])

    return X, Y


def compute_TSNE(self, X: np.ndarray):
    """
    Compute the t-SNE embedding for visualization of high-dimensional data.

    Args:
        self: Object with `self.idx` attribute (typically a client object).
        X (np.ndarray): Input features to be embedded.

    Returns:
        np.ndarray: 2D t-SNE embedded representation of the data.
    """
    print("Computing TSNE of client {0}".format(self.idx))
    np.random.seed(25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(X)
    return embedded_data
