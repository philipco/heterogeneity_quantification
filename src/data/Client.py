import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn

from src.optim.Train import train_neural_network, evaluate_test_metric


class Client:

    def __init__(self, train_features, test_features, train_labels, test_labels, output_dim: int, net: nn.Module, criterion, metric, step_size: int):
        super().__init__()
        self.test_size = 0.2
        self.input_dim = train_features.shape[1]
        self.output_dim = output_dim

        # Type of network to use, simply a class
        self.net = net()
        self.criterion = criterion
        self.metric = metric
        self.step_size = step_size

        self.train_features, self.test_features = train_features, test_features
        self.train_labels, self.test_labels = train_labels, test_labels

        # self.projecteur = features @ torch.linalg.pinv(features.T @ features) @ features.T

    def resplit_train_test(self):
        self.train_features, self.test_features, self.train_labels, self.test_labels \
            = train_test_split(torch.concat([self.train_features, self.test_features]),
                               torch.concat([self.train_labels, self.test_labels]), test_size=self.test_size)

    def train(self, nb_epochs: int, batch_size: int):
        criterion = self.criterion()
        self.trained_model, self.train_loss = train_neural_network(self.net, self.train_features, self.train_labels,
                                                                   criterion, nb_epochs, self.step_size, batch_size)
        # plt.plot(self.train_loss)
        # plt.show()


        # Compute test metrics
        test_metric = evaluate_test_metric(self.trained_model, self.test_features, self.test_labels, self.metric)
        print(f"Test metric:", test_metric)
        # Compute test loss
        test_outputs = self.trained_model(self.test_features)
        self.test_loss = criterion(test_outputs, self.test_labels)
        atomic_criterion = self.criterion(reduction='none')
        self.atomic_test_losses = atomic_criterion(test_outputs, self.test_labels)


