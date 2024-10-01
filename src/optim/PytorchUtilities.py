import copy
from typing import List

import numpy as np
import torch

from src.data.Network import Network
from src.quantif.Metrics import Metrics


def get_models_of_collaborating_models(network: Network, test_quantile: Metrics, i: int):
    pvalue_matrix = test_quantile.aggreage_heter()
    models, weights = [], []
    for j in range(network.nb_clients):
        models.append(network.clients[j].trained_model)
        # if i == j:
        #     weights.append(len(network.clients[j].Y_train))
        if pvalue_matrix[i][j] > 0.05 or i == j:
            weights.append(len(network.clients[j].train_loader.dataset))
        else:
            weights.append(0)
        # We always add the model of the client itself.

    return models, weights / np.sum(weights)


def print_collaborating_clients(network: Network, test_quantile: Metrics):
    pvalue_matrix = test_quantile.aggreage_heter()
    list_of_collaboration = {}
    for i in range(network.nb_clients):
        list_of_collaboration[i] = []
        for j in range(network.nb_clients):
            if pvalue_matrix[i][j] > 0.05 or i == j:
                list_of_collaboration[i].append(j)
    print(list_of_collaboration)

def aggregate_models(models: List[torch.nn.Module], weights: List[int]) -> torch.nn.Module:
    # Assume models is a list of PyTorch models of the same architecture

    # Initialize the parameters of the first model as the running sum
    avg_model = copy.deepcopy(models[0])

    # Loop through the other models and add their parameters to the first model
    with torch.no_grad():  # Disable gradient tracking
        for name, param in avg_model.named_parameters():
            # Initialize the parameter tensor to zero
            avg_param = torch.zeros_like(param)
            # Sum the parameters from all models
            for model, weight in zip(models, weights):
                avg_param += (model.state_dict()[name].clone() * weight)

            # Update the averaged model with the new parameter values
            avg_model.state_dict()[name].copy_(avg_param)
    return avg_model