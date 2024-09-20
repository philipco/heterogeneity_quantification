import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn

from src.optim.Train import train_local_neural_network, evaluate_test_metric


class Client:

    def __init__(self, ID, X_train, X_val, X_test, Y_train, Y_val, Y_test, output_dim: int, net: nn.Module,
                 criterion, metric, step_size: int, momentum: int):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {self.device}")

        self.ID = ID

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test.to(self.device)
        self.Y_train, self.Y_val, self.Y_test = Y_train, Y_val, Y_test.to(self.device)

        self.input_dim = self.X_train.shape[1]
        self.output_dim = output_dim

        # Type of network to use, simply a class
        self.net = net
        self.criterion = criterion
        self.metric = metric
        self.step_size, self.momentum = step_size, momentum
        self.last_epoch = 0

        # self.projecteur = features @ torch.linalg.pinv(features.T @ features) @ features.T

    def resplit_train_test(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test \
            = train_test_split(torch.concat([self.X_train, self.X_test]),
                               torch.concat([self.Y_train, self.Y_test]), test_size=self.test_size)



    def train(self, nb_epochs: int, batch_size: int):
        criterion = self.criterion()

        self.trained_model, self.train_loss, self.writer \
            = train_local_neural_network(self.net(), self.device, self.ID, self.X_train, self.X_val,
                                                                         self.Y_train, self.Y_val, criterion, nb_epochs,
                                                                         self.step_size, self.momentum, batch_size,
                                                                         self.metric, self.last_epoch)
        self.last_epoch += nb_epochs

        # Compute test metrics
        test_metric = evaluate_test_metric(self.trained_model, self.X_test, self.Y_test, self.metric)
        print(f"\nTest metric:", test_metric)
        # Compute test loss
        test_outputs = self.trained_model(self.X_test)
        self.test_loss = criterion(test_outputs, self.Y_test)
        atomic_criterion = self.criterion(reduction='none')
        self.atomic_test_losses = atomic_criterion(test_outputs, self.Y_test)


    def continue_training(self, nb_epochs: int, batch_size: int):
        criterion = self.criterion()

        self.trained_model, self.train_loss, self.writer \
            = train_local_neural_network(self.trained_model, self.device, self.ID, self.X_train, self.X_val,
                                                                         self.Y_train, self.Y_val, criterion, nb_epochs,
                                                                         self.step_size, self.momentum, batch_size,
                                                                         self.metric, self.last_epoch, self.writer)
        self.last_epoch += nb_epochs

        # Compute test metrics
        test_metric = evaluate_test_metric(self.trained_model, self.X_test, self.Y_test, self.metric)
        print(f"\nTest metric:", test_metric)
        # Compute test loss
        test_outputs = self.trained_model(self.X_test)
        self.test_loss = criterion(test_outputs, self.Y_test)
        atomic_criterion = self.criterion(reduction='none')
        self.atomic_test_losses = atomic_criterion(test_outputs, self.Y_test)


