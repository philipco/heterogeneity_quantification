import copy
import time
from copy import deepcopy
from random import sample

import numba
import numpy as np
import optuna
import torch
from sklearn.preprocessing import normalize

from src.data.Network import Network
from src.optim.PytorchUtilities import print_collaborating_clients, aggregate_models, \
    get_models_of_collaborating_models, equal, load_new_model, aggregate_gradients, fednova_aggregation
from src.optim.Train import compute_loss_and_accuracy, update_model, gradient_step, \
    write_train_val_test_performance, write_grad, compute_gradient_validation_set
from src.plot.PlotDistance import plot_pvalues
from src.quantif.Distances import compute_matrix_of_distances, acceptance_pvalue, \
    rejection_pvalue
from src.quantif.Metrics import Metrics
from src.utils.Utilities import set_seed


def loss_accuracy_central_server(network: Network, weights, writer, epoch):
    # On the training set.
    epoch_train_loss, epoch_train_accuracy = 0, 0
    for i in range(network.nb_clients):
        client = network.clients[i]
        # WARNING : For tcga_brca, we need to evaluate the metric on the full dataset.
        loss, acc = compute_loss_and_accuracy(client.trained_model, client.device, client.train_loader,
                                              client.criterion, client.metric, "tcga_brca" in client.ID)
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
                                              client.criterion, client.metric, "tcga_brca" in client.ID)
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
                                              client.criterion, client.metric, "tcga_brca" in client.ID)
        epoch_test_loss += loss * weights[i]
        epoch_test_accuracy += acc * weights[i]

    writer.add_scalar('test_loss', epoch_test_loss, epoch)
    writer.add_scalar('test_accuracy', epoch_test_accuracy, epoch)

    writer.add_scalar(f'generalisation_loss', abs(epoch_train_loss - epoch_test_loss), epoch)
    writer.add_scalar(f'generalisation_accuracy', abs(epoch_train_accuracy - epoch_test_accuracy), epoch)

    writer.close()


def federated_training(network: Network, nb_of_synchronization: int = 5, nb_of_local_epoch: int = 1):

    total_nb_points = np.sum([len(client.train_loader.dataset) for client in network.clients])
    weights = [len(client.train_loader.dataset) / total_nb_points for client in network.clients]
    loss_accuracy_central_server(network, weights, network.writer, network.nb_initial_epochs)

    # Averaging models
    new_model = aggregate_models([client.trained_model for client in network.clients],
                     weights, network.clients[0].device)

    for i in range(network.nb_clients):
        load_new_model(network.clients[i].trained_model, new_model)
        assert equal(network.clients[i].trained_model, network.clients[0].trained_model), \
            (f"Models {network.clients[i].ID} are not equal.")

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"=============== \tEpoch {synchronization_idx} ===============")
        start_time = time.time()
        # One pass of local training
        for  i in range(network.nb_clients):
            network.clients[i].continue_training(nb_of_local_epoch, synchronization_idx, single_batch=False)

        # Averaging models
        new_model = aggregate_models([client.trained_model for client in network.clients],
                         weights, network.clients[0].device)
        for client in network.clients:
            load_new_model(client.trained_model, new_model)
            assert equal(client.trained_model, network.clients[0].trained_model), \
                (f"Models 0 and {network.clients[i].ID} are not equal.")
            write_train_val_test_performance(client.trained_model, client.device, client.train_loader,
                                             client.val_loader, client.test_loader, client.criterion, client.metric,
                                             client.ID, client.writer, client.last_epoch)
        loss_accuracy_central_server(network, weights, network.writer, client.last_epoch)
        print(f"Elapsed time: {time.time() - start_time} seconds")



def fednova_training(network: Network, nb_of_synchronization: int = 5, nb_of_local_epoch: int = 1):

    total_nb_points = np.sum([len(client.train_loader.dataset) for client in network.clients])
    weights = [len(client.train_loader.dataset) / total_nb_points for client in network.clients]
    loss_accuracy_central_server(network, weights, network.writer, network.nb_initial_epochs)

    # Number of local epoch * number of samples / batch size
    tau_i = [np.floor(nb_of_local_epoch * len(client.train_loader.dataset)
                        / len(list(client.train_loader)[0])) for client in network.clients]

    # Averaging models
    new_model = aggregate_models([client.trained_model for client in network.clients],
                     weights, network.clients[0].device)

    for i in range(network.nb_clients):
        load_new_model(network.clients[i].trained_model, new_model)
        assert equal(network.clients[i].trained_model, network.clients[0].trained_model), \
            (f"Models {network.clients[i].ID} are not equal.")

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"=============== \tEpoch {synchronization_idx} ===============")
        start_time = time.time()
        old_model = deepcopy(network.clients[0].trained_model)
        # One pass of local training
        for  i in range(network.nb_clients):
            network.clients[i].continue_training(nb_of_local_epoch, synchronization_idx, single_batch=False)

        # Averaging models
        new_model = fednova_aggregation(old_model, [client.trained_model for client in network.clients], tau_i, weights,
                            network.clients[0].device)

        for client in network.clients:
            load_new_model(client.trained_model, new_model)
            assert equal(client.trained_model, network.clients[0].trained_model), \
                (f"Models 0 and {network.clients[i].ID} are not equal.")
            write_train_val_test_performance(client.trained_model, client.device, client.train_loader,
                                             client.val_loader, client.test_loader, client.criterion, client.metric,
                                             client.ID, client.writer, client.last_epoch)
        loss_accuracy_central_server(network, weights, network.writer, client.last_epoch)
        print(f"Elapsed time: {time.time() - start_time} seconds")


def compute_weight_based_on_gradient(gradients, validation_gradient, norm_validation_gradient, local, client_idx):
    grad_norm_diff = []
    n = len(gradients)
    for _ in range(n):
        grads_G = [p.flatten() for p in gradients[_] if p is not None]
        grad_norm_diff.append(
            torch.sum(torch.cat([(gF - gG).pow(2) for gF, gG in zip(grads_G, validation_gradient)]))
        )

    weight = [float(d < norm_validation_gradient) for d in grad_norm_diff]
    if sum(weight) == 0 or local:
        return [int(i == client_idx) for i in range(n)]

    total_weight = sum(weight)
    return [w / total_weight for w in weight]


def all_for_one_algo(network: Network, nb_of_synchronization: int = 5, inner_iterations: int = 50,
                     plot_matrix: bool = True, pruning: bool = False, logs="light", local: bool=False,
                     collab_based_on_grad=False):
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    total_nb_points = np.sum([len(client.train_loader.dataset) for client in network.clients])
    fed_weights = [len(client.train_loader.dataset) / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, fed_weights, network.writer, network.nb_initial_epochs)

    metrics_for_comparison = Metrics(f"{network.dataset_name}_{network.algo_name}",
                              "_comparaison", network.nb_clients, network.nb_testpoints_by_clients)

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"===============\tEpoch {synchronization_idx}\t===============")
        start_time = time.time()

        inner_iterations = max([len(c.train_loader) for c in network.clients])

        # We create a data iterator for each clients, for each trained model by clients.
        iter_loaders = [[iter(client.train_loader) for client in network.clients] for client in network.clients]

        # We compute the validation gradients at the beginning of the inner iterations.
        validation_gradients, norm_validation_gradients = [], []
        for c in network.clients:
            g = compute_gradient_validation_set(c.val_loader, c.trained_model, c.device,
                                                       c.optimizer, c.criterion)
            validation_gradients.append([p.flatten() for p in g])
            norm_validation_gradients.append(torch.sum(torch.cat([gF.pow(2) for gF in validation_gradients[-1]])))

        for k in range(inner_iterations):
            weights = []

            # Compute the new model client by client.
            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]

                gradients = []

                # Computing gradients on each clients.
                for c_idx in range(network.nb_clients):
                    c = network.clients[c_idx]
                    set_seed(client.last_epoch * inner_iterations + k)

                    # Restarting the iterator if it has reached the end.
                    if (c.last_epoch * inner_iterations + k) % len(c.train_loader) == 0:
                        iter_loaders[client_idx][c_idx] = iter(c.train_loader)

                    # Single batch update before aggregation.
                    # TODO : probleme with the scheduler.
                    gradient = gradient_step(iter_loaders[client_idx][c_idx], c.device, client.trained_model,
                                             client.criterion, client.optimizer, client.scheduler)

                    gradients.append(gradient)

                if collab_based_on_grad:
                    weight = compute_weight_based_on_gradient(gradients, validation_gradients[client_idx],
                                                           norm_validation_gradients[client_idx], local, client_idx)
                else:
                    metrics_for_comparison.reinitialize()
                    compute_matrix_of_distances(rejection_pvalue, network, metrics_for_comparison, symetric_distance=True)
                    weight = [int(client_idx == j or metrics_for_comparison.aggreage_heter()[client_idx][j] > 0.05) for j in range(network.nb_clients)]

                    weight = [w / sum(weight) for w in weight]

                weights.append(weight)

                aggregated_gradients = aggregate_gradients(gradients, weight, client.device)
                update_model(client.trained_model, aggregated_gradients, client.optimizer)
        for client in network.clients:
            client.last_epoch += 1
            write_train_val_test_performance(client.trained_model, client.device, client.train_loader,
                                             client.val_loader, client.test_loader, client.criterion, client.metric,
                                             client.ID, client.writer, client.last_epoch)
        for client in network.clients:
            client.scheduler.step()

        print(f"Elapsed time: {time.time() - start_time} seconds")
        print("Now computing matrix of distances...")

        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)

        # The network has trial parameter only if the pruning is active (for hyperparameters search).
        if pruning:
            if network.trial.should_prune():
                raise optuna.TrialPruned()

        if synchronization_idx % 2 == 1:
            ### We compute the distance between clients.
            metrics_for_comparison.reinitialize()
            metrics_for_comparison.add_distances(weights)
            plot_pvalues(metrics_for_comparison, f"{synchronization_idx}")


def all_for_all_algo(network: Network, nb_of_synchronization: int = 5, inner_iterations: int = 50,
                     plot_matrix: bool = True, pruning: bool = False, logs="light", local: bool=False):
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    total_nb_points = np.sum([len(client.train_loader.dataset) for client in network.clients])
    fed_weights = [len(client.train_loader.dataset) / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, fed_weights, network.writer, network.nb_initial_epochs)

    acceptance_test = Metrics(f"{network.dataset_name}_{network.algo_name}",
                              "acceptance_test", network.nb_clients, network.nb_testpoints_by_clients)

    rejection_test = Metrics(f"{network.dataset_name}_{network.algo_name}",
                             "rejection_test", network.nb_clients, network.nb_testpoints_by_clients)

    compute_matrix_of_distances(acceptance_pvalue, network,
                                acceptance_test, symetric_distance=False)
    compute_matrix_of_distances(rejection_pvalue, network,
                                rejection_test, symetric_distance=True)

    if plot_matrix:
        plot_pvalues(acceptance_test, f"{0}")
        plot_pvalues(rejection_test, f"{0}")

    nb_collaborations = [0 for client in network.clients]

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"===============\tEpoch {synchronization_idx}\t===============")
        start_time = time.time()

        # Compute weights len(client.train_loader.dataset) / total_nb_points
        if local:
            weights = [[int(i == j) for j in range(network.nb_clients)] for i in range(network.nb_clients)]
        else:
            weights = [[int(i == j or rejection_test.aggreage_heter()[i][j] > 0.05) for j in range(network.nb_clients)]
                                 for i in range(network.nb_clients)]
            weights = normalize(weights, axis=1, norm='l1')
            weights = weights @ weights.T

        inner_iterations = max([len(c.train_loader) for c in network.clients])
        iter_loaders = [iter(client.train_loader) for client in network.clients]
        for k in range(inner_iterations):

            nb_collaborations = [nb_collaborations[i] + np.sum(weights[i] != 0) for i in range(network.nb_clients)]
            for (client, c) in zip(network.clients, nb_collaborations):
                client.writer.add_scalar('nb_collaborations', c, client.last_epoch)

            gradients = []

            # Computing gradients on each clients.
            for c_idx in range(network.nb_clients):
                client = network.clients[c_idx]
                set_seed(client.last_epoch * inner_iterations + k)

                # Restarting the iterator if it has reached the end.
                if (client.last_epoch * inner_iterations + k) % len(client.train_loader) == 0:
                    iter_loaders[c_idx] = iter(client.train_loader)

                # Single batch update before aggregation.
                gradient = gradient_step(iter_loaders[c_idx], client.device, client.trained_model,
                                      client.criterion, client.optimizer, client.scheduler)

                gradients.append(gradient)

            # Averaging gradient and updating models.
            for i in range(network.nb_clients):
                client = network.clients[i]
                # We plot the computed gradient on each client before their aggregation.
                if logs=="full":
                    write_grad(client.trained_model, client.writer, client.writer)

                aggregated_gradients = aggregate_gradients(gradients, weights[i], client.device)
                update_model(client.trained_model, aggregated_gradients, client.optimizer)

        for client in network.clients:
            client.last_epoch += 1
            write_train_val_test_performance(client.trained_model, client.device, client.train_loader,
                                             client.val_loader, client.test_loader, client.criterion, client.metric,
                                             client.ID, client.writer, client.last_epoch)
        for client in network.clients:
            client.scheduler.step()

        print(f"Elapsed time: {time.time() - start_time} seconds")
        print("Now computing matrix of distances...")

        rejection_test.reinitialize()
        compute_matrix_of_distances(rejection_pvalue, network, rejection_test, symetric_distance=True)

        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)

        # The network has trial parameter only if the pruning is active (for hyperparameters search).
        if pruning:
            if network.trial.should_prune():
                raise optuna.TrialPruned()

        if synchronization_idx % 10 == 0:
            ### We compute the distance between clients.
            acceptance_test.reinitialize()
            compute_matrix_of_distances(acceptance_pvalue, network, acceptance_test, symetric_distance=False)

            # rejection_test.reinitialize()
            # compute_matrix_of_distances(rejection_pvalue, network, rejection_test, symetric_distance=True)

            plot_pvalues(acceptance_test, f"{synchronization_idx}")
            plot_pvalues(rejection_test, f"{synchronization_idx}")


def gossip_training(network: Network, nb_of_communication: int = 501):
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
            new_model = aggregate_models([network.clients[idx1].trained_model, network.clients[idx2].trained_model],
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


def fedquantile_training(network: Network, nb_of_local_epoch: int = 5, nb_of_communication: int = 51):

    total_nb_points = np.sum([len(client.train_loader.dataset) for client in network.clients])
    weights = [len(client.train_loader.dataset) / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, weights, network.writer, network.nb_initial_epochs)

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
            weights = [int(i == j) for j in range(network.nb_clients)]
            print(f"For client {i}, weights are: {weights}.")
            # Aggregate directly into model i.
            aggregate_models(i, models_of_collaborating_clients, weights, network.clients[0].device)

        # One pass of local training
        for client in network.clients:
            client.continue_training(nb_of_local_epoch, epoch)

        loss_accuracy_central_server(network, weights, network.writer, epoch * nb_of_local_epoch +1)

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

