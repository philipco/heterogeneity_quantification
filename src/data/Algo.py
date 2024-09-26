from typing import List

import numpy as np
import torch

from src.data.Network import Network
from src.plot.PlotDistance import plot_pvalues
from src.quantif.Distances import function_to_compute_quantile_pvalue, compute_matrix_of_distances
from src.quantif.Metrics import Metrics

def get_models_of_collaborating_models(network: Network, test_quantile: Metrics, i: int):
    pvalue_matrix = test_quantile.aggreage_heter()
    models, weights = [], []
    for j in range(network.nb_clients):
        if pvalue_matrix[i][j] > 0.05 and i != j:
            models.append(network.clients[j].trained_model)
            weights.append(len(network.clients[j].Y_train))
        # We always add the model of the client itself.
        if i == j:
            models.append(network.clients[j].trained_model)
            weights.append(len(network.clients[j].Y_train))
    return models, weights / np.sum(weights)


def print_collaborating_clients(network: Network, test_quantile: Metrics):
    pvalue_matrix = test_quantile.aggreage_heter()
    list_of_collaboration = {}
    for i in range(network.nb_clients):
        list_of_collaboration[i] = []
        for j in range(network.nb_clients):
            if pvalue_matrix[i][j] > 0.05 and i != j:
                list_of_collaboration[i].append(j)
    print(list_of_collaboration)

def aggregate_models(models: List[torch.nn.Module], weights: List[int]) -> torch.nn.Module:
    # Assume models is a list of PyTorch models of the same architecture
    num_models = len(models)

    # Initialize the parameters of the first model as the running sum
    aggregated_model = models[0]

    # Loop through the other models and add their parameters to the first model
    with torch.no_grad():  # Disable gradient tracking
        for name, param in aggregated_model.named_parameters():
            i = 0
            param.data *= weights[i]
            # Sum the parameters from all models
            for model in models[1:]:
                i+=1
                param.data += (model.state_dict()[name].data * weights[i])

        # # Divide by the number of models to compute the average
        # for name, param in aggregated_model.named_parameters():
        #     param.data /= num_models

    return aggregated_model


def federated_training(network: Network, nb_of_local_epoch: int = 1, nb_of_communication: int = 100):

    for round in range(nb_of_communication):
        total_nb_points = np.sum([len(client.Y_train) for client in network.clients])
        central_model = aggregate_models([client.trained_model for client in network.clients],
                                         [len(client.Y_train) / total_nb_points for client in network.clients])

        # Averaging models
        for i in range(network.nb_clients):
            network.clients[i].trained_model = central_model

        # One pass of local training
        for client in network.clients:
            client.continue_training(nb_of_local_epoch, network.batch_size)

def fedquantile_training(network: Network, test_quantile: Metrics,
                       nb_of_local_epoch: int = 1, nb_of_communication: int = 100):

    for round in range(1, nb_of_communication+1):

        print_collaborating_clients(network, test_quantile)

        # Averaging models
        for i in range(network.nb_clients):
            models_of_collaborating_clients, weights = get_models_of_collaborating_models(network, test_quantile, i)
            network.clients[i].net = aggregate_models(models_of_collaborating_clients, weights)

        # One pass of local training
        for client in network.clients:
            client.continue_training(nb_of_local_epoch, network.batch_size)


        ### We compute the distance between clients.
        compute_matrix_of_distances(function_to_compute_quantile_pvalue, network,
                                    test_quantile, symetric_distance=False)


        if round%10 == 0:
            plot_pvalues(test_quantile, f"heter_{round}")

    # Plot test loss!





