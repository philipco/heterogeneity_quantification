from copy import deepcopy
from random import sample

import numpy as np
from dask.array import reciprocal
from torch.utils.tensorboard import SummaryWriter

from src.data.Network import Network
from src.optim.PytorchUtilities import print_collaborating_clients, aggregate_models, \
    get_models_of_collaborating_models, are_identical, copy_a_into_b
from src.optim.Train import compute_loss_and_accuracy
from src.plot.PlotDistance import plot_pvalues
from src.quantif.Distances import function_to_compute_quantile_pvalue, compute_matrix_of_distances
from src.quantif.Metrics import Metrics

def loss_accuracy_central_server(network: Network, weights, writer, epoch):
    # On the training set.
    epoch_train_loss, epoch_train_accuracy = 0, 0
    for i in range(network.nb_clients):
        client = network.clients[i]
        # WARNING : For tcga_brca, we need to evaluate the metric on the full dataset.
        loss, acc = compute_loss_and_accuracy(client.trained_model, client.device, client.train_loader,
                                              client.criterion(), client.metric, "tcga_brca" in client.ID)
        epoch_train_loss += loss * weights[i]
        epoch_train_accuracy += acc * weights[i]

    writer.add_scalar('train_loss', epoch_train_loss, epoch)
    writer.add_scalar('train_accuracy', epoch_train_accuracy, epoch)

    # On the validation set.
    epoch_val_loss, epoch_val_accuracy = 0, 0
    for i in range(network.nb_clients):
        client = network.clients[i]
        # WARNING : For tcga_brca, we need to evaluate the metric on the full dataset.
        loss, acc = compute_loss_and_accuracy(client.trained_model, client.device, client.val_loader,
                                              client.criterion(), client.metric, "tcga_brca" in client.ID)
        epoch_val_loss += loss * weights[i]
        epoch_val_accuracy += acc * weights[i]

    writer.add_scalar('val_loss', epoch_val_loss, epoch)
    writer.add_scalar('val_accuracy', epoch_val_accuracy, epoch)
    writer.close()


def federated_training(network: Network, nb_of_local_epoch: int = 5, nb_of_communication: int = 101):
    writer = SummaryWriter(
        log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{network.algo_name}_{network.dataset_name}_'
                f'central_server')

    total_nb_points = np.sum([len(client.train_loader.dataset) for client in network.clients])
    weights = [len(client.train_loader.dataset) / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, weights, writer, 0)

    for epoch in range(1, nb_of_communication+1):

        aggregate_models(0, [client.trained_model for client in network.clients],
                         weights, network.clients[0].device)

        # Averaging models
        for i in range(1, network.nb_clients):
            network.clients[i].load_new_model(network.clients[0].trained_model)

        # One pass of local training
        for  i in range(network.nb_clients):
            client = network.clients[i]
            client.continue_training(nb_of_local_epoch, network.batch_size, epoch)

        loss_accuracy_central_server(network, weights, writer, epoch * nb_of_local_epoch)

            # # WARNING : For tcga_brca, we need to evaluate the metric on the full dataset.
            # loss, acc = compute_loss_accuracy(client.trained_model, client.device, client.val_loader,
            #                                   client.criterion(), client.metric, "tcga_brca" in client.ID)
            # epoch_val_loss += loss * weights[i]
            # epoch_val_accuracy += acc * weights[i]

        # writer.add_scalar('val_loss', epoch_val_loss, epoch)
        # writer.add_scalar('val_accuracy', epoch_val_accuracy, epoch)

    # Close the writer at the end of training



def gossip_training(network: Network, test_quantile: Metrics,
                         nb_of_local_epoch: int = 1, nb_of_communication: int = 1001):

    nb_clients = network.nb_clients
    sender = np.random.randint(nb_clients)
    for epoch in range(1, nb_of_communication + 1):
        receiver = sample([i for i in range(sender)] + [i for i in range(sender+1, network.nb_clients)], 1)[0]

        max_p, max_lambda = 0, 0
        for lbda in [0.5]:
            _, p = function_to_compute_quantile_pvalue(network, receiver, sender, lbda)
            if p > max_p:
                max_p, max_lambda = p, lbda
        if max_p > 0.25:
            # print(f"Weight sender: {network.clients[sender].trained_model.state_dict()['linear.bias']}")
            # print(f"Weight receiver: {network.clients[receiver].trained_model.state_dict()['linear.bias']}")
            # n_loc, n_rem = len(network.clients[receiver].train_loader.dataset), len(network.clients[sender].train_loader.dataset)
            weights = [max_lambda, 1 - max_lambda]
            print(f"=============== Aggregation for lambda = {max_lambda} and p = {max_p}")
            aggregate_models(0, [network.clients[receiver].trained_model, network.clients[sender].trained_model],
                             weights, network.clients[0].device)
            # print(f"Weight receiver after: {network.clients[receiver].trained_model.state_dict()['linear.bias']}")
        print(f"=============== Epoch {epoch} potentially updates client {receiver} with data from {sender}.")
        network.clients[receiver].continue_training(5, network.batch_size, epoch)

        # The updated client becomes the sender.
        sender = receiver

        if epoch % 100 == 0:
            ### We compute the distance between clients.
            compute_matrix_of_distances(function_to_compute_quantile_pvalue, network,
                                        test_quantile, symetric_distance=False)
            plot_pvalues(test_quantile, f"heter_{epoch}")


def fedquantile_training(network: Network, test_quantile: Metrics,
                         nb_of_local_epoch: int = 5, nb_of_communication: int = 101):
    writer = SummaryWriter(
        log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{network.algo_name}_{network.dataset_name}_'
                f'central_server')

    total_nb_points = np.sum([len(client.train_loader.dataset) for client in network.clients])
    weights = [len(client.train_loader.dataset) / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, weights, writer, 0)

    for epoch in range(1, nb_of_communication+1):
        print(f"=============== Epoch {epoch}")

        print_collaborating_clients(network, test_quantile)

        # Averaging models
        for i in range(network.nb_clients):
            models_of_collaborating_clients, weights = get_models_of_collaborating_models(network, test_quantile, i)
            print(f"For client {i}, weights are: {weights}.")
            # Aggregate directly into model i.
            aggregate_models(i, models_of_collaborating_clients, weights, network.clients[0].device)

        # One pass of local training
        for client in network.clients:
            client.continue_training(nb_of_local_epoch, network.batch_size, epoch)

        loss_accuracy_central_server(network, weights, writer, epoch * nb_of_local_epoch )

        ### We compute the distance between clients.
        test_quantile = Metrics(network.dataset_name, "TEST_QUANTILE", network.nb_clients,
                network.nb_testpoints_by_clients)

        compute_matrix_of_distances(function_to_compute_quantile_pvalue, network,
                                    test_quantile, symetric_distance=False)

        if epoch%5 == 0:
            plot_pvalues(test_quantile, f"heter_{epoch}")
