import copy
from typing import List

import numpy as np
import torch

from src.data.Network import Network
from src.quantif.Metrics import Metrics


def are_identical(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def get_models_of_collaborating_models(network: Network, test_quantile: Metrics, i: int):
    pvalue_matrix = test_quantile.aggreage_heter()
    models, weights = [], []
    for j in range(network.nb_clients):
        models.append(network.clients[j].trained_model)
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


def copy_a_into_b(a: torch.nn.Module, b: torch.nn.Module, device):
    with torch.no_grad():  # Disable gradient tracking
        for name, param in a.named_parameters():
            b.state_dict()[name].copy_(a.state_dict()[name].data.clone())


def aggregate_models(idx_main_model: int, models: List[torch.nn.Module], weights: List[int], device: str) -> torch.nn.Module:
    # Assume models is a list of PyTorch models of the same architecture
    # The main model is the running sum
    # Loop through the other models and add their parameters to the running model
    with torch.no_grad():  # Disable gradient tracking
        for name, param in models[idx_main_model].named_parameters():
            temp_param = torch.zeros(param.shape).to(device)
            for weight, model in zip(weights, models):
                temp_param += weight * model.state_dict()[name].data.clone()

            models[idx_main_model].state_dict()[name].copy_(temp_param)