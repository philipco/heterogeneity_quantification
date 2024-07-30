from src.data.Client import Client
from src.data.DatasetConstants import CRITERION, NB_LABELS, MODELS, STEP_SIZE, METRIC
from src.utils.PickleHandler import pickle_loader, pickle_saver
from src.utils.Utilities import get_project_root, file_exist, create_folder_if_not_existing


class Network:

    def __init__(self, features, labels, batch_size, nb_epochs, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.nb_clients = len(labels)
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.nb_testpoints_by_clients = [len(y[1]) for y in labels]
        self.criterion = CRITERION[dataset_name]

        # Creating clients.
        self.clients = []
        for i in range(self.nb_clients):
            self.clients.append(Client(features[i][0], features[i][1], labels[i][0], labels[i][1],
                                       NB_LABELS[dataset_name], MODELS[dataset_name], CRITERION[dataset_name],
                                       METRIC[dataset_name], STEP_SIZE[dataset_name]))

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