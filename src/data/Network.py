import copy

from transformers import AutoModelForMultipleChoice

from src.data.Client import Client
from src.data.DatasetConstants import CRITERION, MODELS, STEP_SIZE, METRIC, MOMENTUM, BATCH_SIZE, SCHEDULER_PARAMS, \
    WEIGHT_DECAY, CHECKPOINT
from src.utils.LoggingWriter import LoggingWriter
from src.utils.PickleHandler import pickle_loader
from src.utils.Utilities import get_project_root, file_exist, create_folder_if_not_existing, set_seed


class Network:

    def __init__(self, train_loaders, val_loaders, test_loaders, dataset_name,
                 algo_name, split_type, seed=0):
        super().__init__()
        self.trial = None
        set_seed(seed)
        self.dataset_name = dataset_name
        self.split_type = split_type
        self.algo_name = algo_name
        self.nb_clients = len(train_loaders)
        # The iterable dataset has no lenght (online setting, lenght is infinite).
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
                mu = train_loaders[i].dataset.mu
                step_size = 1 / (2 * L)
            elif dataset_name == "synth_complex":
                L = train_loaders[i].dataset.L
                mu = train_loaders[i].dataset.mu
                step_size = 1 / (8 * L)
            self.clients.append(Client(ID, f"{dataset_name}", train_loaders[i], val_loaders[i],
                                       test_loaders[i], copy.deepcopy(net),
                                       CRITERION[dataset_name], METRIC[dataset_name], step_size,
                                       MOMENTUM[dataset_name], WEIGHT_DECAY[dataset_name], BATCH_SIZE[dataset_name],
                                       SCHEDULER_PARAMS[dataset_name]))

        ID = f"{dataset_name}_{algo_name}_central_server" if split_type is None \
            else f"{dataset_name}_{split_type}_{algo_name}_central_server"
        self.writer = LoggingWriter(
            log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{dataset_name}/{ID}')

    @classmethod
    def loader(cls, dataset_name):
        root = get_project_root()
        if file_exist("{0}/pickle/{1}/processed_data/network.pkl".format(root, dataset_name)):
            return pickle_loader("{0}/pickle/{1}/processed_data/network".format(root, dataset_name))
        raise FileNotFoundError()

    def count_trainable_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save(self):
        root = get_project_root()
        pickle_folder = '{0}/pickle/{1}/{2}'.format(root, self.dataset_name, self.algo_name)
        create_folder_if_not_existing(pickle_folder)
        self.writer.save(f"{pickle_folder}", "logging_writer_central.pkl")
        for client in self.clients:
            client.writer.save(f"{pickle_folder}", f"logging_writer_{client.ID}.pkl")