import copy
import time

import numpy as np
import optuna
import torch

from src.data.Network import Network
from src.optim.PytorchUtilities import aggregate_models, equal, load_new_model, aggregate_gradients, fednova_aggregation
from src.optim.Train import compute_loss_and_accuracy, update_model, safe_gradient_computation
from src.utils.Utilities import print_mem_usage


def loss_accuracy_central_server(network: Network, weights, writer, epoch):
    # On the training set.
    epoch_train_loss, epoch_train_accuracy = 0, 0
    for i in range(network.nb_clients):
        client = network.clients[i]
        # WARNING : For tcga_brca, we need to evaluate the metric on the full dataset.
        loss, acc = compute_loss_and_accuracy(client.trained_model, client.device, client.train_loader,
                                              client.criterion, client.metric, "tcga_brca" in client.ID or "synth" in client.ID)
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
                                              client.criterion, client.metric, True)
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
                                              client.criterion, client.metric, True)
        epoch_test_loss += loss * weights[i]
        epoch_test_accuracy += acc * weights[i]

    writer.add_scalar('test_loss', epoch_test_loss, epoch)
    writer.add_scalar('test_accuracy', epoch_test_accuracy, epoch)

    writer.add_scalar(f'generalisation_loss', abs(epoch_train_loss - epoch_test_loss), epoch)
    writer.add_scalar(f'generalisation_accuracy', abs(epoch_train_accuracy - epoch_test_accuracy), epoch)

    writer.close()


def federated_training(network: Network, nb_of_synchronization: int = 5, keep_track: bool = False):

    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    weights = [client.nb_train_points / total_nb_points for client in network.clients]

    try:
        inner_iterations = int(np.mean([len(client.train_loader) for client in network.clients]))
    except TypeError:
        inner_iterations = 1
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    loss_accuracy_central_server(network, weights, network.writer, network.nb_initial_epochs)
    for client in network.clients:
        client.write_train_val_test_performance()

    # Averaging models
    new_model = aggregate_models([client.trained_model for client in network.clients],
                     weights, network.clients[0].device)

    for i in range(network.nb_clients):
        load_new_model(network.clients[i].trained_model, new_model)

    if keep_track:
        track_models = [[[m.data[0].to("cpu") for m in c.trained_model.parameters()]] for c in network.clients]
        # track_gradients = [[] for c in network.clients]

    # We create a data iterator for each clients, for each trained model by clients.
    iter_loaders = [iter(client.train_loader) for client in network.clients]

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"=============== \tEpoch {synchronization_idx} ===============")
        start_time = time.time()
        # One pass of local training
        for i in range(network.nb_clients):
            iter_loaders[i] = network.clients[i].continue_training(inner_iterations, iter_loaders[i])

        # Averaging models
        new_model = aggregate_models([client.trained_model for client in network.clients],
                         weights, network.clients[0].device)

        for client_idx in range(network.nb_clients):
            client = network.clients[client_idx]
            load_new_model(client.trained_model, new_model)
            assert equal(client.trained_model, network.clients[0].trained_model), \
                (f"Models 0 and {network.clients[i].ID} are not equal.")
            client.write_train_val_test_performance()
            if keep_track:
                track_models[client_idx].append([m.data[0].to("cpu") for m in client.trained_model.parameters()])
        loss_accuracy_central_server(network, weights, network.writer, client.last_epoch)
        print(network.writer.retrieve_information('train_loss')[1][-1])
        print(network.writer.retrieve_information('train_accuracy')[1][-1])
        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")
        print_mem_usage()
        network.save()
    if keep_track:
        return track_models
    return None

def fednova_training(network: Network, nb_of_synchronization: int = 5, nb_of_local_epoch: int = 1, keep_track: bool = False):

    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    weights = [client.nb_train_points / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, weights, network.writer, network.nb_initial_epochs)
    for client in network.clients:
        client.write_train_val_test_performance()

    # Number of local epoch * number of samples / batch size
    tau_i = [np.floor(nb_of_local_epoch * client.nb_train_points
                        / client.nb_train_points) for client in network.clients]

    # Averaging models
    new_model = aggregate_models([client.trained_model for client in network.clients],
                     weights, network.clients[0].device)

    for i in range(network.nb_clients):
        load_new_model(network.clients[i].trained_model, new_model)
        assert equal(network.clients[i].trained_model, network.clients[0].trained_model), \
            (f"Models {network.clients[i].ID} are not equal.")

    if keep_track:
        track_models = [[[m.data[0].to("cpu") for m in c.trained_model.parameters()]] for c in network.clients]

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"=============== \tEpoch {synchronization_idx} ===============")
        start_time = time.time()
        old_model = copy.deepcopy(network.clients[0].trained_model)
        # One pass of local training
        for  i in range(network.nb_clients):
            network.clients[i].continue_training(nb_of_local_epoch, synchronization_idx, single_batch=False)

        # Averaging models
        new_model = fednova_aggregation(old_model, [client.trained_model for client in network.clients], tau_i, weights,
                            network.clients[0].device)

        for client_idx in range(network.nb_clients):
            client = network.clients[client_idx]
            load_new_model(client.trained_model, new_model)
            assert equal(client.trained_model, network.clients[0].trained_model), \
                (f"Models 0 and {network.clients[i].ID} are not equal.")
            client.write_train_val_test_performance()
            if keep_track:
                track_models[client_idx].append([m.data[0].to("cpu") for m in client.trained_model.parameters()])
        loss_accuracy_central_server(network, weights, network.writer, client.last_epoch)
        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")
        network.save()
    if keep_track:
        return track_models
    return None

def ratio(x,y):
    if x == 0:
        return 1
    if y == 0:
        return 0
    else:
        if not 1 - x/y <= 1:
            print(f"Not lower than one: {1 - x/y}, x={x}, y={y}.")
        return max(1 - x/y, 0)

def compute_weight_based_on_ratio(gradients, nb_points_by_clients, client_idx, numerators, denominators,
                                  continuous: bool = False) :
    grads = []
    for _ in range(len(gradients)):
        grads.append(torch.concat([p.flatten() for p in gradients[_] if p is not None]))

    new_denom = grads[client_idx].T @ grads[client_idx]
    denominators[client_idx].append(new_denom.item())

    for _ in range(len(gradients)):
        new_num = torch.linalg.vector_norm(grads[client_idx] - grads[_])
        numerators[client_idx][_].append(new_num.item())

    if continuous:
        weight = [ratio(numerators[client_idx][_][-1], denominators[client_idx][-1]) for _ in range(len(gradients))]
    else:
        weight = [int(ratio(numerators[client_idx][_][-1], denominators[client_idx][-1]) > 0) for _ in range(len(gradients))]
        nb_points_by_selected_clients = sum([n if w != 0 else 0 for (w, n) in zip(weight, nb_points_by_clients)])
        weight = [w * n / nb_points_by_selected_clients for  (w, n) in zip(weight, nb_points_by_clients)]

    if weight[client_idx] == 0:
        print(f"⚠️ The client {client_idx} do not collaborate with himself..")

    if sum(weight) == 0:
        print(f"⚠️ The sum of weights is zero.")
        weight = [int(i == client_idx) for i in range(len(gradients))]
    total_weight = sum(weight)
    return [w / total_weight for w in weight], numerators, denominators

def all_for_one_algo(network: Network, nb_of_synchronization: int = 5, continuous: bool = False, pruning: bool = False, keep_track=False):
    try:
        inner_iterations = int(np.mean([len(client.train_loader) for client in network.clients]))
    except TypeError:
        inner_iterations = 1
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    fed_weights = [client.nb_train_points / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, fed_weights, network.writer, network.nb_initial_epochs)
    for client in network.clients:
        client.write_train_val_test_performance()

    numerators, denominators = [[[0] for _ in network.clients] for _ in network.clients], [[0] for _ in network.clients]
    if keep_track:
        track_models = [[[copy.deepcopy(m.data[0]).to("cpu") for m in c.trained_model.parameters()]] for c in network.clients]
        track_gradients = [[] for c in network.clients]

    # We create a data iterator for each clients, for each trained model by clients.
    iter_loaders = [[iter(client.train_loader) for client in network.clients] for client in network.clients]

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"===============\tEpoch {synchronization_idx}\t===============")
        start_time = time.time()

        weights = {_: None for _ in range(network.nb_clients)}
        for client_idx in range(network.nb_clients):
            gradients_eval = {_: None for _ in range(network.nb_clients)}
            client = network.clients[client_idx]
            for c_idx in range(network.nb_clients):
                c = network.clients[c_idx]

                gradient_eval, iter_loaders[client_idx][c_idx] = safe_gradient_computation(c.train_loader, iter_loaders[client_idx][c_idx], c.device,
                                                      client.trained_model, client.criterion, client.optimizer,
                                                      client.scheduler)
                gradients_eval[c_idx] = gradient_eval

            weight, numerators, denominators = compute_weight_based_on_ratio(gradients_eval, network.nb_testpoints_by_clients,
                                                                             client_idx, numerators,
                                                                             denominators, continuous=continuous)

            weights[client_idx] = weight
            network.clients[client_idx].writer.add_histogram('weights', np.array(weight),
                                                             network.clients[client_idx].last_epoch * inner_iterations)

        for k in range(inner_iterations):

            # Compute the new model client by client.
            grad_time = time.time()
            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]

                gradients = []

                for c_idx in range(network.nb_clients):
                    c = network.clients[c_idx]

                    gradient, iter_loaders[client_idx][c_idx] = safe_gradient_computation(c.train_loader, iter_loaders[client_idx][c_idx], c.device,
                                                           client.trained_model, client.criterion, client.optimizer,
                                                           client.scheduler)
                    gradients.append(gradient)

                aggregated_gradients = aggregate_gradients(gradients, weights[client_idx], client.device)
                update_model(client.trained_model, aggregated_gradients, client.optimizer)

                if keep_track:
                    lr = client.optimizer.param_groups[0]['lr']
                    track_models[client_idx].append([copy.deepcopy(m.data[0]).to("cpu") for m in client.trained_model.parameters()])
                    track_gradients[client_idx].append([lr * g[0].to("cpu") for g in aggregated_gradients])
            print(f"Gradients computation time: {time.time() - grad_time} seconds")

        perf_time = time.time()
        for i in range(network.nb_clients):
            client = network.clients[i]
            client.last_epoch += 1
            client.write_train_val_test_performance()

        for client in network.clients:
            client.scheduler.step()

        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)

        print(f"Performance time: {time.time() - perf_time} seconds")

        network.save()
        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")
        print_mem_usage()

        # The network has trial parameter only if the pruning is active (for hyperparameters search).
        if pruning:
            if network.trial.should_prune():
                raise optuna.TrialPruned()

    if keep_track:
        return track_models, track_gradients

def all_for_all_algo(network: Network, nb_of_synchronization: int = 5, pruning: bool = False,
                     keep_track=False, collab_based_on="loss"):
    try:
        inner_iterations = int(np.mean([len(client.train_loader) for client in network.clients]))
    except TypeError:
        inner_iterations = 1
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    fed_weights = [client.nb_train_points / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, fed_weights, network.writer, network.nb_initial_epochs)
    for client in network.clients:
        client.write_train_val_test_performance()

    numerators, denominators = [[[] for _ in network.clients] for _ in network.clients], [[] for _ in network.clients]

    if keep_track:
        track_models = [[[m.data[0].to("cpu") for m in c.trained_model.parameters()]] for c in network.clients]
        track_gradients = [[] for c in network.clients]

    iter_loaders = [iter(client.train_loader) for client in network.clients]

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"===============\tEpoch {synchronization_idx}\t===============")
        start_time = time.time()

        weights = {_: None for _ in range(network.nb_clients)}
        if collab_based_on == "local":
            for client_idx in range(network.nb_clients):
                weight = [int(j == client_idx) for j in range(network.nb_clients)]
                weights[client_idx] = weight
                network.clients[client_idx].writer.add_histogram('weights', np.array(weight),
                                                                 network.clients[client_idx].last_epoch * inner_iterations)

        elif collab_based_on == "ratio":
            gradients_eval = {_: None for _ in range(network.nb_clients)}
            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]

                gradient_eval, iter_loaders[client_idx] = safe_gradient_computation(client.train_loader, iter_loaders[client_idx], client.device,
                                                      client.trained_model, client.criterion, client.optimizer,
                                                      client.scheduler)
                gradients_eval[client_idx] = gradient_eval

            for client_idx in range(network.nb_clients):
                weight, numerators, denominators = compute_weight_based_on_ratio(gradients_eval,
                                                                                 network.nb_testpoints_by_clients,
                                                                                 client_idx, numerators,
                                                                                 denominators, False)
                weights[client_idx] = weight
                network.clients[client_idx].writer.add_histogram('weights', np.array(weight), network.clients[client_idx].last_epoch * inner_iterations)
        else:
            raise ValueError("Collaboration criterion '{0}' is not recognized.".format(collab_based_on))

        for k in range(inner_iterations):

            gradients = []

            # Computing gradients on each clients.
            grad_time = time.time()
            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]

                gradient, iter_loaders[client_idx] = safe_gradient_computation(client.train_loader, iter_loaders[client_idx], client.device,
                                                     client.trained_model, client.criterion, client.optimizer,
                                                     client.scheduler)

                gradients.append(gradient)

            print(f"Gradients computation time: {time.time() - grad_time} seconds")

            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]

                aggregated_gradients = aggregate_gradients(gradients, weights[client_idx], client.device)
                update_model(client.trained_model, aggregated_gradients, client.optimizer)

                if keep_track:
                    lr = client.optimizer.param_groups[0]['lr']
                    track_models[client_idx].append([m.data[0].to("cpu") for m in client.trained_model.parameters()])
                    track_gradients[client_idx].append([lr * g[0].to("cpu") for g in aggregated_gradients])
        perf_time = time.time()
        for client in network.clients:
            client.last_epoch += 1
            client.write_train_val_test_performance()
        for client in network.clients:
            client.scheduler.step()

        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)

        print(f"Performance time: {time.time() - perf_time} seconds")

        network.save()

        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")
        print_mem_usage()

        # The network has trial parameter only if the pruning is active (for hyperparameters search).
        if pruning:
            if network.trial.should_prune():
                raise optuna.TrialPruned()

    if keep_track:
        return track_models, track_gradients

