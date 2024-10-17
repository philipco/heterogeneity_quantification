"""Created by Constantin Philippenko, 29th September 2022."""
import sys

import pandas as pd
import torchvision

from src.data.Dataset import do_prediction_liquid_asset, prepare_liquid_asset, load_liquid_dataset_test
from src.optim.Algo import fedquantile_training, federated_training, gossip_training
from src.data.Network import Network
from src.data.DatasetConstants import NB_CLIENTS, BATCH_SIZE, TRANSFORM_TRAIN, TRANSFORM_TEST
from src.data.DataLoader import get_data_from_pytorch, get_data_from_flamby, get_data_from_csv, get_synth_data
from src.plot.PlotDistance import plot_pvalues
from src.quantif.Metrics import Metrics
from src.utils.Utilities import get_project_root, get_path_to_datasets, set_seed

root = get_project_root()
FLAMBY_PATH = '{0}/../FLamby'.format(root)

sys.path.insert(0, FLAMBY_PATH)
sys.path.insert(0, FLAMBY_PATH + '/flamby')

from datasets.fed_heart_disease.dataset import FedHeartDisease
from datasets.fed_tcga_brca.dataset import FedTcgaBrca
from flamby.datasets.fed_ixi import FedIXITiny

DATASET = {"mnist": torchvision.datasets.MNIST, "cifar10": torchvision.datasets.CIFAR10,
           "heart_disease": FedHeartDisease, "tcga_brca": FedTcgaBrca, "ixi": FedIXITiny
           }

# from datasets.fed_isic2019.dataset import FedIsic2019
# from datasets.fed_tcga_brca.dataset import FedTcgaBrca

NB_RUN = 1


# algo_name = "quantile" # quantile, fed ou gossip
# dataset_name = "tcga_brca"

if __name__ == '__main__':

    for dataset_name in ["mnist"]:
        assert dataset_name in ["mnist", "cifar10", "heart_disease", "tcga_brca", "ixi", "liquid_asset"], \
            "Dataset not recognized."
        print(f"### ================== DATASET: {dataset_name} ================== ###")
        nb_of_clients = NB_CLIENTS[dataset_name]

        nb_initial_epochs = 1 if dataset_name in ["mnist", "cifar10"] else 1

        for algo_name in ["quantile", "gossip"]:
            assert algo_name in ["quantile", "gossip", "fed"], "Algorithm not recognized."
            print(f"--- ================== ALGO: {algo_name} ================== ---")
            ### We the dataset naturally splitted or not.
            if dataset_name in ["mnist", "cifar10"]:
                split_type = "partition"
                X_train, X_val, X_test, Y_train, Y_val, Y_test, natural_split \
                    = get_data_from_pytorch(DATASET[dataset_name], nb_of_clients, split_type,
                                            kwargs_train_dataset = dict(root=get_path_to_datasets(), download=False,
                                                                  transform=TRANSFORM_TRAIN[dataset_name]),
                                            kwargs_test_dataset=dict(root=get_path_to_datasets(), download=False,
                                                                      transform=TRANSFORM_TEST[dataset_name]),
                                            kwargs_dataloader = dict(batch_size=BATCH_SIZE[dataset_name], shuffle=False))
            elif dataset_name in ["liquid_asset"]:
                X_train, X_val, X_test, Y_train, Y_val, Y_test, natural_split \
                    = get_data_from_csv(dataset_name)
            elif dataset_name in ["synth"]:
                X_train, X_val, X_test, Y_train, Y_val, Y_test, natural_split \
                    = get_synth_data(dataset_name)

            else:
                X_train, X_val, X_test, Y_train, Y_val, Y_test, natural_split \
                    = get_data_from_flamby(DATASET[dataset_name], nb_of_clients, dataset_name,
                                           kwargs_dataloader=dict(batch_size=BATCH_SIZE[dataset_name], shuffle=False))

            network = Network(X_train, X_val, X_test, Y_train, Y_val, Y_test, BATCH_SIZE[dataset_name],
                              nb_initial_epochs, dataset_name, algo_name)
            network.save()

            if algo_name == "quantile":
                fedquantile_training(network)
            if algo_name == "gossip":
                gossip_training(network)
            if algo_name == "fed":
                federated_training(network)

        if dataset_name in ["liquid_asset"]:
            X_raw_train, X_raw_test, numerical_transformer = load_liquid_dataset_test(get_path_to_datasets())
            do_prediction_liquid_asset(network, X_raw_train, X_raw_test, numerical_transformer)
