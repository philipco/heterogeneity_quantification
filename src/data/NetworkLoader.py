import sys

import torchvision

from src.data.DataLoader import get_data_from_pytorch, get_data_from_flamby, get_synth_data, get_data_from_csv, \
    get_data_from_NLP
from src.data.DatasetConstants import BATCH_SIZE, TRANSFORM_TEST, TRANSFORM_TRAIN, NB_CLIENTS
from src.data.Network import Network
from src.utils.Utilities import get_project_root, get_path_to_datasets

root = get_project_root()
FLAMBY_PATH = '{0}/../FLamby'.format(root)

sys.path.insert(0, FLAMBY_PATH)

from flamby.datasets.fed_heart_disease.dataset import FedHeartDisease
from flamby.datasets.fed_tcga_brca.dataset import FedTcgaBrca
from flamby.datasets.fed_ixi import FedIXITiny

# from flamby.datasets.fed_isic2019.dataset import FedIsic2019


DATASET = {"mnist": torchvision.datasets.MNIST, "cifar10": torchvision.datasets.CIFAR10,
           "heart_disease": FedHeartDisease, "tcga_brca": FedTcgaBrca, "ixi": FedIXITiny
           }


def get_network(dataset_name: str, algo_name: str, nb_initial_epochs: int):
    split_type = None

    ### We the dataset naturally splitted or not.
    if dataset_name in ["mnist", "cifar10"]:
        split_type = "partition"
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_data_from_pytorch(DATASET[dataset_name], NB_CLIENTS[dataset_name], split_type, BATCH_SIZE[dataset_name],
                                    kwargs_train_dataset=dict(root=get_path_to_datasets(), download=True,
                                                              transform=TRANSFORM_TRAIN[dataset_name]),
                                    kwargs_test_dataset=dict(root=get_path_to_datasets(), download=True,
                                                             transform=TRANSFORM_TEST[dataset_name]),
                                    kwargs_dataloader=dict(batch_size=BATCH_SIZE[dataset_name], shuffle=False))
    elif dataset_name in ["exam_llm"]:
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_data_from_NLP(BATCH_SIZE[dataset_name])
    elif dataset_name in ["liquid_asset"]:
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_data_from_csv(dataset_name, BATCH_SIZE[dataset_name])
    elif dataset_name in ["synth"]:
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_synth_data(BATCH_SIZE[dataset_name])
    elif dataset_name in ["synth_complex"]:
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_synth_data(BATCH_SIZE[dataset_name], nb_clients=6, nb_clusters=2, dim=100)
    else:
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_data_from_flamby(DATASET[dataset_name], NB_CLIENTS[dataset_name], dataset_name, BATCH_SIZE[dataset_name],
                                   kwargs_dataloader=dict(batch_size=BATCH_SIZE[dataset_name], shuffle=False))
    return Network(train_loaders, val_loaders, test_loaders,
                      nb_initial_epochs, dataset_name, algo_name, split_type)