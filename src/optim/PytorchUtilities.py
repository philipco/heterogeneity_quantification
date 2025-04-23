import copy
from copy import deepcopy
from typing import List

import numpy as np
import torch

from src.data.Network import Network
from src.quantif.Metrics import Metrics

def get_models_of_collaborating_models(network: Network, acceptance_test: Metrics, rejection_test: Metrics, i: int):
    acceptance_pvalue = acceptance_test.aggreage_heter()
    rejection_pvalue = rejection_test.aggreage_heter()
    weights = []
    for j in range(network.nb_clients):
        if i == j:
            weights.append(len(network.clients[j].train_loader.dataset))
        else:
            if acceptance_pvalue[i][j] > 0.1 and rejection_pvalue[i][j] > 0.1:
                weights.append(len(network.clients[j].train_loader.dataset))
            else:
                weights.append(0)
        # We always add the model of the client itself.

    return [network.clients[j].trained_model for j in range(network.nb_clients)], weights / np.sum(weights)


def print_collaborating_clients(network: Network, acceptance_test: Metrics, rejection_test: Metrics):
    acceptance_pvalue = acceptance_test.aggreage_heter()
    rejection_pvalue = rejection_test.aggreage_heter()
    list_of_collaboration = {}
    for i in range(network.nb_clients):
        list_of_collaboration[i] = []
        for j in range(network.nb_clients):
            if (acceptance_pvalue[i][j] > 0.1 and rejection_pvalue[i][j] > 0.1) or i == j:
                list_of_collaboration[i].append(j)
    print(list_of_collaboration)


def equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def scalar_multiplication(model, scalar):
    new_model = deepcopy(model)
    for name, param in model.named_parameters():
        new_model.state_dict()[name].copy_(scalar * param)
    return new_model


def load_new_model(model_to_update: torch.nn.Module, new_model: torch.nn.Module) -> None:
    """Updates the parameters of `model_to_update` with the parameters of `new_model`.
    This function replaces each parameter in `model_to_update` with the corresponding parameter from `new_model`.

    :param model_to_update:
        The model whose parameters will be updated.
    :type model_to_update: torch.nn.Module
    :param new_model:
        The model from which the parameters will be copied. It should have the same architecture and parameters as `model_to_update`.
    :type new_model: torch.nn.Module

    :returns:
        None

    **Example usage:**

    .. code-block:: python

        model_to_update = MyModel()
        new_model = MyModel()
        load_new_model(model_to_update, new_model)

    **Note:**
    - The function disables gradient tracking with `torch.no_grad()` to ensure that parameter updates do not interfere
      with the training process.
    - The function assumes that `model_to_update` and `new_model` have identical architectures.
    """
    with torch.no_grad():
        for name, param in model_to_update.named_parameters():
            model_to_update.state_dict()[name].copy_(new_model.state_dict()[name].data.clone())


def aggregate_models(models: List[torch.nn.Module], weights: List[int], device: str) \
        -> torch.nn.Module:
    """Aggregates the parameters of multiple PyTorch models by weighted summation and return it.

    This function updates the parameters of the main model by summing the weighted parameters of all models provided.
    It assumes all models share the same architecture.

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
        device = 'cuda'

        aggregated_model = aggregate_models(models, weights, device)

    **Note:**
    - The function disables gradient tracking with `torch.no_grad()` to ensure that the updates to the parameters do not
    interfere with the training process.
    - The function assumes that the models are already moved to the correct device.
"""

    # Assume models is a list of PyTorch models of the same architecture
    # model_copy save the running sum
    # Loop through the other models and add their parameters to the running model
    model_copy = copy.deepcopy(models[0])
    with torch.no_grad():  # Disable gradient tracking
        for name, param in model_copy.named_parameters():
            temp_param = torch.zeros(param.shape).to(device)
            for weight, model in zip(weights, models):
                temp_param += (weight * model.state_dict()[name].data.clone())

            model_copy.state_dict()[name].copy_(temp_param)
    return model_copy

def fednova_aggregation(old_model, new_models, tau_i, weights, device):
    tau_eff = np.sum([w * tau for w, tau in zip(weights, tau_i)])

    aggreg_new = aggregate_models(new_models, [tau_eff * w / tau for w, tau in zip(weights, tau_i)], device)
    aggreg_old = aggregate_models([old_model for m in new_models],
                                  [tau_eff * w / tau for w, tau in zip(weights, tau_i)], device)
    final_aggreg = aggregate_models([old_model, aggreg_new, aggreg_old],[1,1,-1], device)
    return final_aggreg


def aggregate_gradients(gradients_list, weights, device):
    """
        Aggregates gradients by applying weights.

        gradients_list: list of lists of gradients (each sublist is the gradients from one source)
        weights: list of weights to apply to each set of gradients
        """
    # Initialize aggregated gradients with zeroed tensors of the same shape as the first set of gradients
    aggregated_gradients = [torch.zeros_like(g).to(device) if (g is not None) else None
                            for g in gradients_list[0]]

    for i, gradients in enumerate(gradients_list):
        for j, grad in enumerate(gradients):
            if grad is not None:
                try:
                    aggregated_gradients[j] += weights[i] * grad.to(aggregated_gradients[j].device)
                except IndexError:
                    print(f"i: {i}")
                    print(f"j: {j}")
                    print(f"weights: {weights}")
                    print(f"len: {len(aggregated_gradients)}")



    return aggregated_gradients