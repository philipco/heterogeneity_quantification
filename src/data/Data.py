"""Created by Constantin Philippenko, 29th September 2022."""
from typing import List

import numpy as np
import torch

from src.data.Client import Client
from src.data.DatasetConstants import MODELS, CRITERION, NB_LABELS, STEP_SIZE
from src.data.Split import iid_split
from src.utils.PickleHandler import pickle_saver, pickle_loader
from src.utils.Utilities import get_project_root, create_folder_if_not_existing, file_exist


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


class Network:

    def __init__(self, X, Y, batch_size, nb_epochs, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.nb_clients = len(Y)
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.nb_points_by_clients = [len(y) for y in Y]
        self.criterion = CRITERION[dataset_name]

        # Creating clients.
        self.clients = []
        for i in range(self.nb_clients):
            self.clients.append(Client(X[i], Y[i], NB_LABELS[dataset_name], MODELS[dataset_name],
                                       CRITERION[dataset_name], STEP_SIZE[dataset_name]))

        # Training all clients
        for client in self.clients:
            client.train(self.nb_epochs, self.batch_size)

    @classmethod
    def loader(cls, dataset_name):
        root = get_project_root()
        if file_exist("{0}/pickle/{1}/processed_data/network.pkl".format(root, dataset_name)):
            return pickle_loader("{0}/pickle/{1}/processed_data/network".format(root, dataset_name))
        raise FileNotFoundError()

    def save(self):
        root = get_project_root()
        create_folder_if_not_existing("{0}/pickle/{1}/processed_data".format(root, self.dataset_name))
        pickle_saver(self, "{0}/pickle/{1}/processed_data/network".format(root, self.dataset_name))

    def retrain_all_clients(self):
        for client in self.clients:
            client.resplit_train_test()
            client.train(self.nb_epochs, self.batch_size)
        # self.all_models.train_all_clients(self.features_iid, self.features_heter, self.labels_iid, self.labels_heter)

    # def resplit_iid(self) -> None:
    #     super().resplit_iid()
    #     self.PCA_fit_iid = [fit_PCA(X, IncrementalPCA(n_components=self.pca_nb_components), None, self.batch_size)
    #                         for X in self.features_iid]


def compute_Y_distribution(labels: List[np.array]) -> np.array:
    nb_labels = len(np.unique(np.concatenate(labels)))
    return [np.array([(l == y).sum() / len(l) for y in range(nb_labels)]) for l in labels]