import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from src.optim.Train import train_neural_network


class Client:

    def __init__(self, features, labels, output_dim: int, net: nn.Module, criterion, step_size: int):
        super().__init__()
        self.test_size = 0.2
        self.input_dim = features.shape[1]
        self.output_dim = output_dim

        # Type of network to use, simply a class
        self.net = net
        self.criterion = criterion
        self.step_size = step_size

        self.train_features, self.test_features, self.train_labels, self.test_labels \
            = train_test_split(features.float(), labels.float(), test_size=self.test_size, random_state=42)
        self.models_iid, self.models_heter = None, None

    def resplit_train_test(self):
        self.train_features, self.test_features, self.train_labels, self.test_labels \
            = train_test_split(torch.concat([self.train_features, self.test_features]),
                               torch.concat([self.train_labels, self.test_labels]), test_size=self.test_size)

    def train(self, nb_epochs: int, batch_size: int):
        criterion = self.criterion()
        self.trained_model = train_neural_network(self.net(self.input_dim, self.output_dim), self.train_features,
                                                  self.train_labels, criterion, nb_epochs, self.step_size,
                                                  batch_size)

        # Compute test loss
        test_outputs = self.trained_model(self.test_features)
        self.test_loss = criterion(test_outputs, self.test_labels)
        atomic_criterion = self.criterion(reduction='none')
        self.atomic_test_losses = atomic_criterion(test_outputs, self.test_labels)


