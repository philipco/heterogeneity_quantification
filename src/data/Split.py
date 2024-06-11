"""Created by Constantin Philippenko, 29th September 2022."""
import warnings
from typing import List

import torch
import numpy as np
from sklearn.manifold import TSNE


def iid_split(data: np.ndarray, labels: np.ndarray, nb_clients: int,
              nb_points_by_non_iid_clients: np.array) -> [List[np.ndarray], List[np.ndarray]]:
    nb_points = data.shape[0]
    X = []
    Y = []
    indices = np.arange(nb_points)
    # np.random.seed()
    np.random.shuffle(indices)
    idx_split = [np.sum(nb_points_by_non_iid_clients[:i]) for i in range(1, nb_clients)]
    split_indices = np.array_split(indices, idx_split)
    for i in range(nb_clients):
        X.append(data[split_indices[i]])
        Y.append(labels[split_indices[i]])
    return X, Y


def create_non_iid_split(features: List[np.array], labels: List[np.array], nb_clients: int,
                         natural_split: bool) -> [List[np.ndarray], List[np.ndarray]]:
    np.random.seed(45)
    return dirichlet_split(features, labels, nb_clients)


def sort_and_partition_split(features: np.array, labels: np.array, nb_clients: int) \
        -> [List[np.ndarray], List[np.ndarray]]:
    unique_labels = np.unique(labels)
    nb_of_split_for_one_label = int(np.ceil(2 * nb_clients / len(unique_labels)))

    # We sort features by labels
    sorted_features = []
    sorted_labels = []
    for label in unique_labels:
        sorted_features.append(features[labels == label])
        sorted_labels.append(labels[labels == label])

    X, Y = [[] for i in range(nb_clients)], [[] for i in range(nb_clients)]
    counter_client = 0
    for i in range(len(unique_labels)):
        size = len(sorted_labels[i])
        features_split = np.split(sorted_features[i], [j * size // nb_of_split_for_one_label for j in range(1, nb_of_split_for_one_label)])
        labels_split = np.split(sorted_labels[i], [j * size // nb_of_split_for_one_label for j in range(1, nb_of_split_for_one_label)])

        for j in range(nb_of_split_for_one_label):
            X[counter_client % nb_clients].append(features_split[j])
            Y[counter_client % nb_clients].append(labels_split[j])
            counter_client += 1

    for idx_client in range(nb_clients):
        X[idx_client] = torch.concat((X[idx_client]))
        Y[idx_client] = torch.concat(Y[idx_client])

    return X, Y


def dirichlet_split(data: np.ndarray, labels: np.ndarray, nb_clients: int, dirichlet_coef: float = 0.5) \
        -> [List[np.ndarray], List[np.ndarray]]:
    nb_labels = len(np.unique(labels)) # Here data is not yet split. Thus nb_labels is correct.
    X = [[] for i in range(nb_clients)]
    Y = [[] for i in range(nb_clients)]
    for idx_label in range(nb_labels):
        proportions = np.random.dirichlet(np.repeat(dirichlet_coef, nb_clients))
        assert round(proportions.sum()) == 1, "The sum of proportion is not equal to 1."

        N = len(labels[labels == idx_label])

        split_indices = [np.sum([int(proportions[k] * N) for k in range(j)]) for j in range(1,nb_clients)]
        features_split = np.split(data[labels == idx_label], split_indices)
        labels_split = np.split(labels[labels == idx_label], split_indices)
        for j in range(nb_clients):
            X[j].append(features_split[j])
            Y[j].append(labels_split[j])

    for idx_client in range(nb_clients):
        X[idx_client] = torch.concat((X[idx_client]))
        Y[idx_client] = torch.concat(Y[idx_client])
    return X, Y


def compute_TSNE(self, X: np.ndarray):
    """Compute the TSNE representation of a dataset."""
    print("Computing TSNE of client {0}".format(self.idx))
    np.random.seed(25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(X)
    return embedded_data