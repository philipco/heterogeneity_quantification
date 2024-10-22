import copy
from random import sample

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.data.Network import Network
from src.optim.PytorchUtilities import print_collaborating_clients, aggregate_models, \
    get_models_of_collaborating_models, are_identical
from src.optim.Train import compute_loss_and_accuracy
from src.plot.PlotDistance import plot_pvalues
from src.quantif.Distances import compute_matrix_of_distances, acceptance_pvalue, \
    rejection_pvalue
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

    # On the test set.
    epoch_test_loss, epoch_test_accuracy = 0, 0
    for i in range(network.nb_clients):
        client = network.clients[i]
        # WARNING : For tcga_brca, we need to evaluate the metric on the full dataset.
        loss, acc = compute_loss_and_accuracy(client.trained_model, client.device, client.test_loader,
                                              client.criterion(), client.metric, "tcga_brca" in client.ID)
        epoch_test_loss += loss * weights[i]
        epoch_test_accuracy += acc * weights[i]

    writer.add_scalar('test_loss', epoch_test_loss, epoch)
    writer.add_scalar('test_accuracy', epoch_test_accuracy, epoch)
    writer.close()


def federated_training(network: Network, nb_of_local_epoch: int = 5, nb_of_communication: int = 101):
    writer = SummaryWriter(
        log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{network.dataset_name}_{network.algo_name}_'
                f'central_server')

    total_nb_points = np.sum([len(client.train_loader.dataset) for client in network.clients])
    weights = [len(client.train_loader.dataset) / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, weights, writer, network.nb_initial_epochs)

    for epoch in range(1, nb_of_communication+1):

        aggregate_models(0, [client.trained_model for client in network.clients],
                         weights, network.clients[0].device)

        # Averaging models
        for i in range(1, network.nb_clients):
            network.clients[i].load_new_model(network.clients[0].trained_model)

        # One pass of local training
        for  i in range(network.nb_clients):
            client = network.clients[i]
            client.continue_training(nb_of_local_epoch, epoch)

        loss_accuracy_central_server(network, weights, writer, epoch * nb_of_local_epoch)


def gossip_training(network: Network, nb_of_communication: int = 101):
    acceptance_test = Metrics(f"{network.dataset_name}_{network.algo_name}",
                              "acceptance_test", network.nb_clients, network.nb_testpoints_by_clients)

    rejection_test = Metrics(f"{network.dataset_name}_{network.algo_name}",
                             "rejection_test", network.nb_clients, network.nb_testpoints_by_clients)

    compute_matrix_of_distances(acceptance_pvalue, network,
                                acceptance_test, symetric_distance=False)
    compute_matrix_of_distances(rejection_pvalue, network,
                                rejection_test, symetric_distance=True)

    plot_pvalues(acceptance_test, f"{0}")
    plot_pvalues(rejection_test, f"{0}")

    for epoch in range(1, nb_of_communication + 1):
        for client in network.clients:
            client.continue_training(1, epoch, single_batch=False)

        # Two clients are sampled
        idx1 = sample([i for i in range(network.nb_clients)], 1)[0]
        idx2 = sample([i for i in range(idx1)] + [i for i in range(idx1+1, network.nb_clients)], 1)[0]

        weights = [0.5, 1 - 0.5]
        p_accept = acceptance_pvalue(network, idx1, idx2, 0.5)
        p_reject = rejection_pvalue(network, idx1, idx2) # Symmetric pvalue
        model = copy.deepcopy(network.clients[idx1].trained_model)
        print(f"=============== \tEpoch {epoch} - p_accept {p_accept}\tp_reject: {p_reject}.")
        if p_accept > 0.1 and p_reject > 0.1:
            print(f"=============== Epoch {epoch} updates client {idx1} with data from {idx2}.")
            aggregate_models(0, [network.clients[idx1].trained_model, network.clients[idx2].trained_model],
                             weights, network.clients[0].device)
            # print(f"Weight receiver after: {network.clients[receiver].trained_model.state_dict()['linear.bias']}")

        p_accept = acceptance_pvalue(network, idx2, idx1, 0.5)
        if p_accept > 0.1 and p_reject > 0.1:
            print(f"=============== Epoch {epoch} updates client {idx2} with data from {idx1}.")
            aggregate_models(0, [network.clients[idx2].trained_model, model],
                             weights, network.clients[0].device)
            # print(f"Weight receiver after: {network.clients[receiver].trained_model.state_dict()['linear.bias']}")

        if epoch % 50 == 0:
            ### We compute the distance between clients.
            acceptance_test.reinitialize()
            rejection_test.reinitialize()

            compute_matrix_of_distances(acceptance_pvalue, network,
                                        acceptance_test, symetric_distance=False)
            compute_matrix_of_distances(rejection_pvalue, network,
                                        rejection_test, symetric_distance=True)

            plot_pvalues(acceptance_test, f"{epoch}")
            plot_pvalues(rejection_test, f"{epoch}")


def fedquantile_training(network: Network, nb_of_local_epoch: int = 5, nb_of_communication: int = 101):
    writer = SummaryWriter(
        log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{network.dataset_name}_{network.algo_name}_'
                f'central_server')

    total_nb_points = np.sum([len(client.train_loader.dataset) for client in network.clients])
    weights = [len(client.train_loader.dataset) / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, weights, writer, network.nb_initial_epochs)

    acceptance_test = Metrics(f"{network.dataset_name}_{network.algo_name}",
                            "acceptance_test", network.nb_clients, network.nb_testpoints_by_clients)

    rejection_test = Metrics(f"{network.dataset_name}_{network.algo_name}",
                              "rejection_test", network.nb_clients, network.nb_testpoints_by_clients)

    compute_matrix_of_distances(acceptance_pvalue, network,
                                acceptance_test, symetric_distance=False)
    compute_matrix_of_distances(rejection_pvalue, network,
                                rejection_test, symetric_distance=True)

    plot_pvalues(acceptance_test, "")
    plot_pvalues(rejection_test, "")

    for epoch in range(1, nb_of_communication+1):
        print(f"=============== Epoch {epoch}")

        print_collaborating_clients(network, acceptance_test, rejection_test)

        # Averaging models
        for i in range(network.nb_clients):
            models_of_collaborating_clients, weights = get_models_of_collaborating_models(network, acceptance_test,
                                                                                          rejection_test, i)
            print(f"For client {i}, weights are: {weights}.")
            # Aggregate directly into model i.
            aggregate_models(i, models_of_collaborating_clients, weights, network.clients[0].device)

        # One pass of local training
        for client in network.clients:
            client.continue_training(nb_of_local_epoch, epoch)

        loss_accuracy_central_server(network, weights, writer, epoch * nb_of_local_epoch +1)

        ### We compute the distance between clients (required at each epoch to decide collaboration or not).
        acceptance_test.reinitialize()
        rejection_test.reinitialize()

        compute_matrix_of_distances(acceptance_pvalue, network,
                                    acceptance_test, symetric_distance=False)
        compute_matrix_of_distances(rejection_pvalue, network,
                                    rejection_test, symetric_distance=True)

        if epoch%10 == 0:
            plot_pvalues(acceptance_test, f"{epoch}")
            plot_pvalues(rejection_test, f"{epoch}")
