import torch.nn as nn
from tqdm import tqdm
from src.optim.Train import train_neural_network

class ModelsOfAllClients:

    def __init__(self, net: nn.Module, criterion, output_dim: int, nb_clients: int):
        self.net = net
        self.criterion = criterion
        self.nb_clients = nb_clients
        self.output_dim = output_dim

    def train_all_clients(self, features_iid, features_heter, labels_iid, labels_heter):
        print("Training networks for each client.")
        self.models_iid, self.test_loss_iid = [], []
        self.models_heter, self.test_loss_heter = [], []
        input_dim = features_iid[0].shape[1]
        for i in tqdm(range(self.nb_clients)):
            net, loss = train_neural_network(self.net(input_dim, self.output_dim), features_iid[i], labels_iid[i],
                                             criterion=self.criterion)
            self.models_iid.append(net)
            self.test_loss_iid.append(loss)
            net, loss = train_neural_network(self.net(input_dim, self.output_dim), features_heter[i],
                                             labels_heter[i], criterion=self.criterion)
            self.models_heter.append(net)

            self.test_loss_heter.append(loss)

class LinearRegression(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # One input feature, one output

    def forward(self, x):
        return self.linear(x).flatten()


class CNNClassifier(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
