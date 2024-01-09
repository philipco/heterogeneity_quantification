"""Created by Constantin Philippenko, 20th April 2022."""
import math
from typing import List

import numpy as np
from numpy import linalg as LA
import ot
from tqdm import tqdm
import torch
import torch.nn as nn

from src.data.Data import Data, DataDecentralized, DataCentralized
from src.quantif.Metrics import Metrics


def compute_KL_distance(distrib1: np.ndarray, distrib2: np.ndarray) -> float:
    """Kullback-Leibler."""
    assert len(distrib1) == len(distrib2), "KL: distributions are not of equal size."
    EPS = 10e-6
    N = len(distrib1)
    return np.sum([distrib1[i] * np.log((distrib1[i] / (distrib2[i]+ EPS)) + EPS) for i in range(N)])


def compute_TV_distance(distrib1: np.ndarray, distrib2: np.ndarray) -> float:
    """Total Variation."""
    assert len(distrib1) == len(distrib2), "TV: distributions are not of equal size."
    N = len(distrib1)
    return 0.5 * np.sum([np.abs(distrib1[i] - distrib2[i]) for i in range(N)])


def sub_sample(distrib, nb_sample):
    if nb_sample == distrib.shape[0]:
        return distrib
    idx = np.random.randint(distrib.shape[0], size=nb_sample)
    return distrib[idx, :]


def compute_EM_distance(distrib1: np.ndarray, distrib2: np.ndarray, stochastic: bool) -> float:
    """Earth's mover."""
    assert len(distrib1) >= 1 and len(distrib2) >= 1, "Distributions must not be empty."
    assert len(distrib1[0]) <= 20, "Dimension is bigger than 20."
    nb_sample1, nb_sample2 = distrib1.shape[0], distrib2.shape[0]

    percent_kept = 0.05 if stochastic else 1
    batch_size1, batch_size2 = int(nb_sample1 * percent_kept), int(percent_kept * nb_sample2)

    minibatch_emd = []

    for k in range(int(2 * percent_kept ** -1)):
        np.random.seed(k)
        sub_distrib1 = sub_sample(distrib1, batch_size1)
        np.random.seed(k)
        sub_distrib2 = sub_sample(distrib2, batch_size2)

        cost_matrix = ot.dist(sub_distrib1, sub_distrib2)
        a, b = ot.unif(len(sub_distrib1)), ot.unif(len(sub_distrib2))  # uniform distribution on samples
        # a = torch.ones(len(sub_distrib1)) / len(sub_distrib1)
        # b = torch.ones(len(sub_distrib2)) / len(sub_distrib2)
        minibatch_emd.append(ot.emd2(a, b, cost_matrix))  # Wasserstein distance / EMD value
    return np.average(minibatch_emd)
    # et si je n'averagais pas ?


def reconstruction_error(estimator, X, y=None):
    return LA.norm(X - estimator.inverse_transform(estimator.transform(X))) / len(X)


def compute_PCA_error(PCA_fitted, distrib: np.ndarray):
    return reconstruction_error(PCA_fitted, distrib)


def compute_LSR_distance(net, test_loss, features, labels):
    criterion = nn.MSELoss()
    with torch.no_grad():
        outputs = net(features)
        remote_loss = criterion(outputs, labels)
    return abs(remote_loss - test_loss) / test_loss


def compute_net_distance(net, criterion, test_loss, features, labels):
    outputs = net(features)
    remote_loss = criterion(outputs, labels)
    return remote_loss - test_loss # TODO : Flexibilité !


def function_to_compute_net_error(data: DataDecentralized, i: int, j: int):
    d_iid = compute_net_distance(data.all_models.models_iid[i], data.all_models.criterion,
                                 data.all_models.test_loss_iid[i], data.features_iid[j], data.labels_iid[j])
    d_heter = compute_net_distance(data.all_models.models_heter[i], data.all_models.criterion,
                                   data.all_models.test_loss_heter[i], data.features_heter[j], data.labels_heter[j])
    return d_iid, d_heter


def function_to_compute_LSR_error(data: DataDecentralized, i: int, j: int):
    d_iid = compute_LSR_distance(data.all_models.models_iid[i], data.all_models.test_loss_iid[i],
                                 data.features_iid[j], data.labels_iid[j])
    d_heter = compute_LSR_distance(data.all_models.models_heter[i], data.all_models.test_loss_heter[i],
                                   data.features_heter[j], data.labels_heter[j])
    return d_iid, d_heter


def function_to_compute_TV_error(data: DataDecentralized, i: int, j: int):
    d_iid = compute_TV_distance(data.labels_iid_distrib[i], data.labels_iid_distrib[j])
    d_heter = compute_TV_distance(data.labels_heter_distrib[i], data.labels_heter_distrib[j])
    return d_iid, d_heter


def function_to_compute_PCA_error(data: DataDecentralized, i: int, j: int):
    d_iid = compute_PCA_error(data.PCA_fit_iid[i], data.features_iid[j])
    d_heter = compute_PCA_error(data.PCA_fit_heter[i], data.features_heter[j])
    return d_iid, d_heter


def function_to_compute_EM(data: DataCentralized, i: int, j: int):
    stochastic_emd = False if max([distrib.shape[0] for distrib in data.features_heter]) <= 1000 else True
    d_iid = compute_EM_distance(data.features_iid[i], data.features_iid[j], stochastic=stochastic_emd)
    d_heter = compute_EM_distance(data.features_heter[i], data.features_heter[j], stochastic=stochastic_emd)
    return d_iid, d_heter


def compute_matrix_of_distances(fonction_to_compute_distance, data: Data, metrics: Metrics,
                                symetric_distance: bool = True) -> Metrics:

    distances_iid = np.zeros((data.nb_clients, data.nb_clients))
    distances_heter = np.zeros((data.nb_clients, data.nb_clients))
    for i in tqdm(range(data.nb_clients)):
        start = i if symetric_distance else 0
        for j in range(start, data.nb_clients):
            d_iid, d_heter = fonction_to_compute_distance(data, i, j)
            distances_iid[i,j] = d_iid
            distances_heter[i, j] = d_heter

            if symetric_distance:
                distances_iid[j, i] = d_iid
                distances_heter[j, i] = d_heter

    metrics.add_distances(distances_iid, distances_heter)
    return metrics
