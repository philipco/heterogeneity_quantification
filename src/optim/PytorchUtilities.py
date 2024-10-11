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

def aggregate_models(idx_main_model: int, models: List[torch.nn.Module], weights: List[int], device: str) -> torch.nn.Module:
    """Aggregates the parameters of multiple PyTorch models by weighted summation. The resulting aggregated model is
    stored in the main model specified by the index `idx_main_model`.

    This function updates the parameters of the main model by summing the weighted parameters of all models provided.
    It assumes all models share the same architecture.

    :param int idx_main_model:
        Index of the main model in the `models` list. This model will be updated with the aggregated weights.
    :param models:
        List of PyTorch models. All models should have the same architecture and parameters.
    :type models: List[torch.nn.Module]
    :param weights:
        List of integer weights corresponding to each model. The parameters of each model will be multiplied by its respective weight before aggregation.
    :type weights: List[int]
    :param device:
        The device on which the computations should be performed (e.g., 'cpu', 'cuda').
    :type device: str

    :returns:
        The updated main model with aggregated parameters.
    :rtype: torch.nn.Module

    **Example usage:**

    .. code-block:: python

        models = [model1, model2, model3]
        weights = [0.2, 0.3, 0.5]
        idx_main_model = 0
        device = 'cuda'

        aggregated_model = aggregate_models(idx_main_model, models, weights, device)

    **Note:**
    - The function disables gradient tracking with `torch.no_grad()` to ensure that the updates to the parameters do not
    interfere with the training process.
    - The function assumes that the models are already moved to the correct device.
"""

    # Assume models is a list of PyTorch models of the same architecture
    # The main model is the running sum
    # Loop through the other models and add their parameters to the running model
    with torch.no_grad():  # Disable gradient tracking
        for name, param in models[idx_main_model].named_parameters():
            temp_param = torch.zeros(param.shape).to(device)
            for weight, model in zip(weights, models):
                temp_param += weight * model.state_dict()[name].data.clone()

            models[idx_main_model].state_dict()[name].copy_(temp_param)