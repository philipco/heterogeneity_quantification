"""Created by Constantin Philippenko, 29th September 2022."""
import sys

import torchvision

from src.data.Data import Network
from src.data.DatasetConstants import NB_CLIENTS, TRANSFORM
from src.data.DataLoader import get_data_from_pytorch, get_data_from_flamby
from src.plot.PlotDistance import plot_distance
from src.quantif.Metrics import Metrics
from src.quantif.Distances import compute_matrix_of_distances, function_to_compute_PCA_error, \
    function_to_compute_EM, function_to_compute_TV_error, function_to_compute_LSR_error, \
    function_to_compute_nn_distance, function_to_compute_ranksums_pvalue
from src.utils.Utilities import get_project_root, get_path_to_datasets



root = get_project_root()
FLAMBY_PATH = '{0}/../../FLamby'.format(root)

sys.path.insert(0, FLAMBY_PATH)
import flamby
sys.path.insert(0, FLAMBY_PATH + '/flamby')
import datasets

from datasets.fed_heart_disease.dataset import FedHeartDisease
from datasets.fed_tcga_brca.dataset import FedTcgaBrca
from flamby.datasets.fed_ixi import FedIXITiny

DATASET = {"mnist": torchvision.datasets.MNIST, "cifar10": torchvision.datasets.CIFAR10,
           "heart_disease": FedHeartDisease, "tcga_brca": FedTcgaBrca, "ixi": FedIXITiny}

# from datasets.fed_isic2019.dataset import FedIsic2019
# from datasets.fed_tcga_brca.dataset import FedTcgaBrca

batch_size = 256
nb_epochs = 250
dataset_name = "heart_disease"
nb_of_clients = NB_CLIENTS[dataset_name]

if __name__ == '__main__':

    ### We the dataset naturally splitted or not.
    if dataset_name in ["mnist", "cifar10"]:
        X, Y, natural_split = get_data_from_pytorch(DATASET[dataset_name], nb_of_clients,
                                                kwargs_dataset = dict(root=get_path_to_datasets(), download=False, transform=TRANSFORM[dataset_name]),
                                                kwargs_dataloader = dict(batch_size=batch_size, shuffle=False))

    else:
        X, Y, natural_split = get_data_from_flamby(DATASET[dataset_name], nb_of_clients,
                                               kwargs_dataloader=dict(batch_size=batch_size, shuffle=False))

    ### We process the data in a decentralized way.
    network = Network(X, Y, batch_size, nb_epochs, dataset_name)
    network.save()

    ### We define three measures : PCA and EM on features, and TV on labels.
    metrics_NET = Metrics(dataset_name, "NET", network.nb_clients, network.nb_points_by_clients)
    metrics_RANKS = Metrics(dataset_name, "RANKS", network.nb_clients, network.nb_points_by_clients)

    for i in range(20):
        print(f"=== RUN {i+1} ===")
        ### We compute the distance between clients.

        compute_matrix_of_distances(function_to_compute_ranksums_pvalue, network, metrics_RANKS, symetric_distance=False)
        compute_matrix_of_distances(function_to_compute_nn_distance, network, metrics_NET, symetric_distance=False)

        ### We need to retrain the client
        network.retrain_all_clients()


    print(metrics_NET.aggreage_heter())
    print(metrics_RANKS.aggreage_heter())

    ### We print the distances.
    plot_distance(metrics_NET)
    plot_distance(metrics_RANKS, "pvalue")
