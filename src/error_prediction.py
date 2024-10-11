"""Created by Constantin Philippenko, 29th September 2022."""
import sys

import torchvision

from src.optim.Algo import fedquantile_training, federated_training, gossip_training
from src.data.Network import Network
from src.data.DatasetConstants import NB_CLIENTS, BATCH_SIZE, TRANSFORM_TRAIN, TRANSFORM_TEST
from src.data.DataLoader import get_data_from_pytorch, get_data_from_flamby, get_data_from_csv, get_synth_data
from src.plot.PlotDistance import plot_pvalues
from src.quantif.Metrics import Metrics
from src.quantif.Distances import compute_matrix_of_distances, function_to_compute_quantile_pvalue
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

    for dataset_name in ["liquid_asset"]:
        nb_of_clients = NB_CLIENTS[dataset_name]

        nb_epochs = 50 if dataset_name in ["mnist", "cifar10"] else 1

        for algo_name in ["quantile", "fed", "gossip"]:
            ### We the dataset naturally splitted or not.
            if dataset_name in ["mnist", "cifar10"]:
                X_train, X_val, X_test, Y_train, Y_val, Y_test, natural_split \
                    = get_data_from_pytorch(DATASET[dataset_name], nb_of_clients,
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

            network = Network(X_train, X_val, X_test, Y_train, Y_val, Y_test, BATCH_SIZE[dataset_name], nb_epochs,
                              dataset_name, algo_name)
            network.save()

            ### We define three measures : PCA and EM on features, and TV on labels.
            metrics_TEST_QUANTILE = Metrics(f"{algo_name}_{dataset_name}" , "TEST_QUANTILE", network.nb_clients,
                                            network.nb_testpoints_by_clients)

            for i in range(NB_RUN):
                print(f"=== RUN {i+1} ===")
                ### We compute the distance between clients.

                compute_matrix_of_distances(function_to_compute_quantile_pvalue, network,
                                            metrics_TEST_QUANTILE, symetric_distance=False)


            print(metrics_TEST_QUANTILE.aggreage_heter())

            ### We print the distances in the heterogeneous scenario.
            plot_pvalues(metrics_TEST_QUANTILE, "heter")

            if algo_name == "quantile":
                fedquantile_training(network, metrics_TEST_QUANTILE)
            if algo_name == "gossip":
                gossip_training(network, metrics_TEST_QUANTILE)
            if algo_name == "fed":
                federated_training(network)

        # do_prediction("X_test.csv")
        # federated_training(network)