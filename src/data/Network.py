from src.data.Client import Client
from src.data.DatasetConstants import CRITERION, NB_LABELS, MODELS, STEP_SIZE, METRIC, MOMENTUM, BATCH_SIZE
from src.utils.PickleHandler import pickle_loader, pickle_saver
from src.utils.Utilities import get_project_root, file_exist, create_folder_if_not_existing


class Network:

    def __init__(self, X_train, X_val, X_test, Y_train, Y_val, Y_test, batch_size, nb_epochs, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.nb_clients = len(Y_train)
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.nb_testpoints_by_clients = [len(y) for y in Y_test]
        print(f"Number of test points by clients: {self.nb_testpoints_by_clients}")
        self.criterion = CRITERION[dataset_name]

        # Creating clients.
        self.clients = []
        for i in range(self.nb_clients):
            self.clients.append(Client(f"{dataset_name}_{i}", X_train[i], X_val[i], X_test[i],
                                       Y_train[i], Y_val[i], Y_test[i],
                                       NB_LABELS[dataset_name], MODELS[dataset_name], CRITERION[dataset_name],
                                       METRIC[dataset_name], STEP_SIZE[dataset_name], MOMENTUM[dataset_name],
                                       BATCH_SIZE[dataset_name]))

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
            # client.resplit_train_test()
            client.train(self.nb_epochs, self.batch_size)