import copy
import sys

import torchvision
from transformers import AutoModelForMultipleChoice

from src.data.Client import Client
from src.data.DatasetConstants import (
    CRITERION, MODELS, STEP_SIZE, METRIC, MOMENTUM, BATCH_SIZE,
    SCHEDULER_PARAMS, WEIGHT_DECAY, CHECKPOINT
)
from src.utils.LoggingWriter import LoggingWriter
from src.utils.PickleHandler import pickle_loader
from src.utils.Utilities import (
    get_project_root, file_exist, create_folder_if_not_existing, set_seed, get_path_to_datasets
)

from src.data.DataLoader import get_data_from_pytorch, get_data_from_flamby, get_synth_data, get_data_from_csv, \
    get_data_from_NLP
from src.data.DatasetConstants import BATCH_SIZE, TRANSFORM_TEST, TRANSFORM_TRAIN, NB_CLIENTS

root = get_project_root()
FLAMBY_PATH = '{0}/../FLamby'.format(root)

sys.path.insert(0, FLAMBY_PATH)

from flamby.datasets.fed_heart_disease.dataset import FedHeartDisease
from flamby.datasets.fed_tcga_brca.dataset import FedTcgaBrca
from flamby.datasets.fed_ixi import FedIXITiny


class Network:
    """
    Represents a federated learning network comprising multiple clients.

    This class initializes clients with their respective datasets and models,
    manages training configurations, and handles logging and checkpointing
    functionalities.
    """

    def __init__(self, train_loaders, val_loaders, test_loaders, dataset_name,
                 algo_name, split_type, seed=0):
        """
        Initialize the Network with data loaders and configuration parameters.

        Args:
            train_loaders (list): List of training data loaders for each client.
            val_loaders (list): List of validation data loaders for each client.
            test_loaders (list): List of test data loaders for each client.
            dataset_name (str): Name of the dataset being used.
            algo_name (str): Name of the algorithm applied.
            split_type (str): Type of data splitting strategy.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
        """
        super().__init__()
        self.trial = None
        set_seed(seed)
        self.dataset_name = dataset_name
        self.split_type = split_type
        self.algo_name = algo_name
        self.nb_clients = len(train_loaders)
        # The iterable dataset has no length (online setting, length is infinite).
        try:
            self.nb_testpoints_by_clients = [len(y.dataset) for y in train_loaders]
        except TypeError:
            self.nb_testpoints_by_clients = [1 for y in val_loaders]
        print(f"Number of test points by clients: {self.nb_testpoints_by_clients}")
        self.criterion = CRITERION[dataset_name]

        # Creating clients.
        self.clients = []
        if dataset_name in ["exam_llm"]:
            net = AutoModelForMultipleChoice.from_pretrained(CHECKPOINT, cache_dir="./")
            # Freeze all pretrained weights
            for param in net.base_model.parameters():
                param.requires_grad = False
        else:
            net = MODELS[dataset_name]()

        d = self.count_trainable_parameters(net)
        print(f"Number of trainable parameters: {d}.")
        step_size = STEP_SIZE[dataset_name]
        for i in range(self.nb_clients):
            ID = f"{dataset_name}_{algo_name}_{i}" if split_type is None \
                else f"{dataset_name}_{split_type}_{algo_name}_{i}"
            if "synth" == dataset_name:
                L = train_loaders[i].dataset.L
                step_size = 1 / (2 * L)
            elif dataset_name == "synth_complex":
                L = train_loaders[i].dataset.L
                step_size = 1 / (8 * L)
            self.clients.append(Client(
                ID, f"{dataset_name}", algo_name, train_loaders[i], val_loaders[i],
                test_loaders[i], net,
                CRITERION[dataset_name], METRIC[dataset_name], step_size,
                MOMENTUM[dataset_name], WEIGHT_DECAY[dataset_name], BATCH_SIZE[dataset_name],
                SCHEDULER_PARAMS[dataset_name]
            ))

        ID = f"{dataset_name}_{algo_name}_central_server" if split_type is None \
            else f"{dataset_name}_{split_type}_{algo_name}_central_server"
        self.writer = LoggingWriter(
            log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{dataset_name}/{ID}'
        )

    @classmethod
    def loader(cls, dataset_name):
        """
        Load a saved Network instance from a pickle file.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            Network: Loaded Network instance.

        Raises:
            FileNotFoundError: If the pickle file does not exist.
        """
        root = get_project_root()
        if file_exist(f"{root}/pickle/{dataset_name}/processed_data/network.pkl"):
            return pickle_loader(f"{root}/pickle/{dataset_name}/processed_data/network")
        raise FileNotFoundError()

    def count_trainable_parameters(self, model):
        """
        Count the number of trainable parameters in a model.

        Args:
            model (torch.nn.Module): The model to inspect.

        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save(self):
        """
        Save the logging writers of the central server and all clients.

        This method saves the state of the logging writers to the appropriate
        pickle files for later retrieval and analysis.
        """
        root = get_project_root()
        pickle_folder = f'{root}/pickle/{self.dataset_name}/{self.algo_name}'
        create_folder_if_not_existing(pickle_folder)
        self.writer.save(f"{pickle_folder}", "logging_writer_central.pkl")
        for client in self.clients:
            client.writer.save(f"{pickle_folder}", f"logging_writer_{client.ID}.pkl")



DATASET = {"mnist": torchvision.datasets.MNIST, "mnist_iid": torchvision.datasets.MNIST,
           "cifar10": torchvision.datasets.CIFAR10, "cifar10_iid": torchvision.datasets.CIFAR10,
           "heart_disease": FedHeartDisease, "tcga_brca": FedTcgaBrca, "ixi": FedIXITiny
           }


def get_network(dataset_name: str, algo_name: str):
    """
    Prepare data loaders according to the dataset and instantiate a Network.

    This function selects the appropriate data loading procedure based on the
    given dataset name. It supports various datasets with different splitting
    strategies such as Dirichlet, IID, NLP, CSV, synthetic datasets, and Flamby.
    After loading the data, it creates and returns a Network instance initialized
    with the prepared data loaders and configuration parameters.

    Args:
        dataset_name (str): Name of the dataset to load.
        algo_name (str): Name of the algorithm to use in the Network.

    Returns:
        Network: Initialized Network object with data loaders and settings.
    """
    split_type = None

    ### We the dataset naturally splitted or not.
    if dataset_name in ["mnist", "cifar10"]:
        split_type = "dirichlet"
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_data_from_pytorch(dataset_name, DATASET[dataset_name], NB_CLIENTS[dataset_name], split_type,
                                    kwargs_train_dataset=dict(root=get_path_to_datasets(), download=True,
                                                              transform=TRANSFORM_TRAIN[dataset_name]),
                                    kwargs_test_dataset=dict(root=get_path_to_datasets(), download=True,
                                                             transform=TRANSFORM_TEST[dataset_name]),
                                    kwargs_dataloader=dict(batch_size=BATCH_SIZE[dataset_name], shuffle=True))
    elif dataset_name in ["mnist_iid", "cifar10_iid"]:
        split_type = "iid"
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_data_from_pytorch(dataset_name, DATASET[dataset_name], NB_CLIENTS[dataset_name], split_type,
                                    kwargs_train_dataset=dict(root=get_path_to_datasets(), download=True,
                                                              transform=TRANSFORM_TRAIN[dataset_name]),
                                    kwargs_test_dataset=dict(root=get_path_to_datasets(), download=True,
                                                             transform=TRANSFORM_TEST[dataset_name]),
                                    kwargs_dataloader=dict(batch_size=BATCH_SIZE[dataset_name], shuffle=True))
    elif dataset_name in ["exam_llm"]:
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_data_from_NLP(BATCH_SIZE[dataset_name])
    elif dataset_name in ["liquid_asset"]:
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_data_from_csv(dataset_name, BATCH_SIZE[dataset_name])
    elif dataset_name in ["synth"]:
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_synth_data(BATCH_SIZE[dataset_name], nb_clients=NB_CLIENTS[dataset_name], nb_clusters=1)
    elif dataset_name in ["synth_complex"]:
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_synth_data(BATCH_SIZE[dataset_name], nb_clients=NB_CLIENTS[dataset_name], nb_clusters=1, dim=10)
    elif dataset_name in ["synth_classif"]:
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_synth_data(BATCH_SIZE[dataset_name], nb_clients=2, nb_clusters=1, dim=10, classification=True)
    else:
        train_loaders, val_loaders, test_loaders, natural_split \
            = get_data_from_flamby(DATASET[dataset_name], NB_CLIENTS[dataset_name], dataset_name, BATCH_SIZE[dataset_name],
                                   kwargs_dataloader=dict(batch_size=BATCH_SIZE[dataset_name], shuffle=True))
    return Network(train_loaders, val_loaders, test_loaders, dataset_name, algo_name, split_type)