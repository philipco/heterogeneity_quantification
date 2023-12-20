"""Created by Constantin Philippenko, 29th September 2022."""
import sys

from src.data.DatasetConstants import TRANSFORM_MIST, NB_CLIENTS
from src.data.DataLoader import get_data_from_pytorch, get_data_from_flamby
from src.data.DataProcessing import decentralized_processing_of_data, centralized_processing_of_data
from src.plot.PlotDistance import plot_distance
from src.quantif.Metrics import Metrics
from src.quantif.Distances import compute_matrix_of_distances, function_to_compute_PCA_error, \
    function_to_compute_EM, function_to_compute_TV_error, function_to_compute_LSR_error, function_to_compute_net_error
from src.utils.Utilities import get_project_root, get_path_to_datasets



root = get_project_root()
FLAMBY_PATH = '{0}/../../FLamby'.format(root)

sys.path.insert(0, FLAMBY_PATH)
import flamby
sys.path.insert(0, FLAMBY_PATH + '/flamby')
import datasets

from datasets.fed_heart_disease.dataset import FedHeartDisease
# from datasets.fed_isic2019.dataset import FedIsic2019
# from datasets.fed_tcga_brca.dataset import FedTcgaBrca

batch_size = 256
dataset_name = "heart_disease"
nb_of_clients = NB_CLIENTS[dataset_name]

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':

    ### We the dataset naturally splitted or not.
    # X, Y, natural_split = get_data_from_pytorch(datasets.MNIST, nb_of_clients,
    #                                             kwargs_dataset = dict(root=get_path_to_datasets(), download=False, transform=TRANSFORM_MIST),
    #                                             kwargs_dataloader = dict(batch_size=batch_size, shuffle=False))

    X, Y, natural_split = get_data_from_flamby(FedHeartDisease, nb_of_clients,
                                               kwargs_dataloader=dict(batch_size=batch_size, shuffle=False))

    ### We process the data in a centralized or decentralized way.
    # data_centralized = centralized_processing_of_data(dataset_name, X, Y, batch_size, nb_of_clients)
    data_decentralized = decentralized_processing_of_data(dataset_name, X, Y, batch_size, nb_of_clients)

    ### We define three measures : PCA and EM on features, and TV on labels.
    metrics_PCA = Metrics(dataset_name, "PCA", data_decentralized.nb_clients, data_decentralized.nb_points_by_clients)
    metrics_LSR = Metrics(dataset_name, "LSR", data_decentralized.nb_clients, data_decentralized.nb_points_by_clients)

    for i in range(5):
        ### We compute the distance between clients.
        # compute_matrix_of_distances(function_to_compute_PCA_error, data_decentralized, metrics_PCA)
        compute_matrix_of_distances(function_to_compute_net_error, data_decentralized, metrics_LSR, symetric_distance=False)

        ### We need to resplit the iid dataset.
        # TODO : il faut aussi ré-entrainer.
        data_decentralized.resplit_iid()
        data_decentralized.retrain_all_clients()

        # data_centralized.resplit_iid()

    print(metrics_LSR.distance_heter[0])

    ### We print the distances.
    # plot_distance(metrics_PCA)
    plot_distance(metrics_LSR)
