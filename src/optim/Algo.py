from typing import List

import numpy as np
import torch

from src.data.Network import Network
from src.optim.PytorchUtilities import print_collaborating_clients, aggregate_models, get_models_of_collaborating_models
from src.plot.PlotDistance import plot_pvalues
from src.quantif.Distances import function_to_compute_quantile_pvalue, compute_matrix_of_distances
from src.quantif.Metrics import Metrics




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
                         nb_of_local_epoch: int = 5, nb_of_communication: int = 100):

    for epoch in range(1, nb_of_communication+1):

        print_collaborating_clients(network, test_quantile)

        # Averaging models
        for i in range(network.nb_clients):
            models_of_collaborating_clients, weights = get_models_of_collaborating_models(network, test_quantile, i)
            print(f"For client {i}, weights are: {weights}.")
            # network.clients[i].trained_model = aggregate_models(models_of_collaborating_clients, weights)



        # One pass of local training
        for client in network.clients:
            for name, param in network.clients[i].trained_model.named_parameters():
                network.clients[i].writer.add_histogram(f'{name}.grad', param.grad, 2 * epoch)
                network.clients[i].writer.add_histogram(f'{name}.weight', param, 2 * epoch)

            client.continue_training(nb_of_local_epoch, network.batch_size)
            
            for name, param in network.clients[i].trained_model.named_parameters():
                network.clients[i].writer.add_histogram(f'{name}.grad', param.grad, 2 * epoch +1)
                network.clients[i].writer.add_histogram(f'{name}.weight', param, 2 * epoch +1)


        ### We compute the distance between clients.
        compute_matrix_of_distances(function_to_compute_quantile_pvalue, network,
                                    test_quantile, symetric_distance=False)


        if epoch%100 == 0:
            plot_pvalues(test_quantile, f"heter_{epoch}")






