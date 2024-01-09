"""Created by Constantin Philippenko, 29th September 2022."""
from typing import List

import numpy as np
import torch
from sklearn.decomposition import IncrementalPCA

from src.data.DatasetConstants import PCA_NB_COMPONENTS, MODELS, CRITERION, NB_LABELS, STEP_SIZE
from src.data.Split import iid_split
from src.optim.nn.Nets import ModelsOfAllClients
from src.utils.UtilitiesNumpy import fit_PCA


class Data:

    def __init__(self, dataset_name: str, nb_points_by_clients: List[int], features_iid: List[np.array],
                 features_heter: List[np.array], labels_iid: List[np.array], labels_heter: List[np.array]) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.nb_points_by_clients = nb_points_by_clients
        self.nb_clients = len(features_heter)

        self.features_iid = features_iid
        self.features_heter = features_heter
        self.labels_iid = labels_iid
        self.labels_heter = labels_heter

        self.labels_iid_distrib = compute_Y_distribution(self.labels_iid)
        self.labels_heter_distrib = compute_Y_distribution(self.labels_heter)

    def resplit_iid(self) -> None:
        nb_points_by_clients = [len(l) for l in self.labels_heter]
        features_iid, labels_iid = iid_split(torch.concat(self.features_heter), torch.concat(self.labels_heter),
                                             self.nb_clients, nb_points_by_clients)

        self.features_iid = features_iid
        self.labels_iid = labels_iid

        self.labels_iid_distrib = compute_Y_distribution(self.labels_iid)


class DataCentralized(Data):

    def __init__(self, dataset_name: str, nb_points_by_clients: List[int], features_iid: List[np.array],
                 features_heter: List[np.array], labels_iid: List[np.array], labels_heter: List[np.array]) -> None:
        super().__init__(dataset_name, nb_points_by_clients, features_iid, features_heter, labels_iid,
                         labels_heter)

    def resplit_iid(self) -> None:
        super().resplit_iid()


class DataDecentralized(Data):

    def __init__(self, dataset_name: str, nb_points_by_clients: List[int], features_iid: List[np.array],
                 features_heter: List[np.array], labels_iid: List[np.array], labels_heter: List[np.array],
                 batch_size: int) -> None:
        super().__init__(dataset_name, nb_points_by_clients, features_iid, features_heter, labels_iid, labels_heter)
        self.batch_size = batch_size
        dim = self.features_heter[0].shape[1]
        self.pca_nb_components = PCA_NB_COMPONENTS if dim > PCA_NB_COMPONENTS else dim // 2
        # print("Fitting decentralized PCA on iid split.")
        # self.PCA_fit_iid = [fit_PCA(X, IncrementalPCA(n_components=self.pca_nb_components), None, self.batch_size)
        #                     for X in self.features_iid]
        # print("Fitting decentralized PCA on heterogeneous split.")
        # self.PCA_fit_heter = [fit_PCA(X, IncrementalPCA(n_components=self.pca_nb_components), None, self.batch_size)
        #                       for X in self.features_heter]

        self.all_models = ModelsOfAllClients(MODELS[dataset_name], CRITERION[dataset_name], NB_LABELS[dataset_name],
                                             self.nb_clients, STEP_SIZE[dataset_name])
        self.all_models.train_all_clients(features_iid, features_heter, labels_iid, labels_heter)

    def retrain_all_clients(self):
        self.all_models.train_all_clients(self.features_iid, self.features_heter, self.labels_iid, self.labels_heter)

    def resplit_iid(self) -> None:
        super().resplit_iid()
        self.PCA_fit_iid = [fit_PCA(X, IncrementalPCA(n_components=self.pca_nb_components), None, self.batch_size)
                            for X in self.features_iid]


def compute_Y_distribution(labels: List[np.array]) -> np.array:
    nb_labels = len(np.unique(np.concatenate(labels)))
    return [np.array([(l == y).sum() / len(l) for y in range(nb_labels)]) for l in labels]