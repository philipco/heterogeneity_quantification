"""
Federated Learning Algorithms: FedAvg, FedNova, All-for-One, All-for-All

This script implements and compares various federated learning (FL) algorithms within a unified framework.
It supports:

- FedAvg: The standard Federated Averaging algorithm.
- FedNova: Federated Normalized Averaging, which corrects client update heterogeneity.
- All-for-One: Clients selectively collaborate by choosing with whom to average their gradients/models.
- All-for-All: Clients consider all peers but apply individual reweighting schemes for model aggregation.

Each algorithm proceeds in communication rounds (synchronization steps). In each round, clients perform local updates
and models are aggregated globally using a specific scheme. The script is designed to track training, validation, and test
performance during training and supports optional tracking of model parameters.
"""

import copy
import gc
import time

import numpy as np
import optuna
import torch
from torch import optim

from src.data.Network import Network
from src.utils.UtilitiesPytorch import aggregate_models, equal, load_new_model, aggregate_gradients, \
    fednova_aggregation, scalar_multiplication
from src.optim.Train import compute_loss_and_accuracy, update_model, safe_gradient_computation, continue_training
from src.utils.Utilities import print_mem_usage


def loss_accuracy_central_server(network: Network, weights, writer, epoch):
    """
   Computes and logs the weighted training, validation, and test loss/accuracy
   of the models stored locally on each client.

   Parameters:
       network (Network): The federated network including all clients and their models.
       weights (list of float): Weights for each client based on number of local training points.
       writer (SummaryWriter): TensorBoard writer for logging metrics.
       epoch (int): Current global epoch (synchronization round).
   """
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


def fedavg_training(network: Network, nb_of_synchronization: int = 5, keep_track: bool = False):
    """
    Federated Averaging (FedAvg) implementation with periodic model synchronization.

    Parameters:
        network (Network): The federated system including all clients and their local data/models.
        nb_of_synchronization (int): Number of communication rounds to run.
        keep_track (bool): Whether to store model parameters at each round (for later analysis/visualization).

    Returns:
        Optional[list]: If `keep_track` is True, returns a list containing tracked model parameters.
    """
    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    weights = [client.nb_train_points / total_nb_points for client in network.clients]

    try:
        inner_iterations = int(np.mean([len(client.train_loader) for client in network.clients]))
    except TypeError:
        inner_iterations = 1
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    loss_accuracy_central_server(network, weights, network.writer, 0)
    for client in network.clients:
        assert equal(client.trained_model, network.clients[0].trained_model), \
            (f"Models 0 and {client.ID} are not equal.")
        client.write_train_val_test_performance()

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
            client = network.clients[i]
            iter_loaders[i] = continue_training(inner_iterations, client.train_loader, iter_loaders[i],
                                                client.trained_model, client.criterion, client.optimizer,
                                                client.device, client.scheduler)
            client.last_epoch += inner_iterations

        # Averaging models
        new_model = aggregate_models([client.trained_model for client in network.clients],
                         weights, network.clients[0].device)

        for client_idx in range(network.nb_clients):
            client = network.clients[client_idx]
            load_new_model(client.trained_model, new_model)
            assert equal(client.trained_model, network.clients[0].trained_model), \
                (f"Models 0 and {client.ID} are not equal.")
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

    torch.cuda.empty_cache()
    gc.collect()
    print_mem_usage("Memory usage at the end of the algo")

    if keep_track:
        return track_models
    return None

def fednova_training(network: Network, nb_of_synchronization: int = 5, nb_of_local_epoch: int = 1, keep_track: bool = False):
    """
    FedNova: Federated Normalized Averaging training loop.

    Parameters:
        network (Network): The federated system including all clients and their local data/models.
        nb_of_synchronization (int): Number of global communication rounds.
        nb_of_local_epoch (int): Number of local epochs for client updates between synchronizations.
        keep_track (bool): Whether to store model parameters after each synchronization.

    Returns:
        Optional[list]: If `keep_track` is True, returns list of tracked model parameters.
    """
    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    weights = [client.nb_train_points / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, weights, network.writer, 0)
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
            # TODO !
            client = network.clients[i]
            continue_training(nb_of_local_epoch, client.train_loader, iter_loaders[i],
                                                client.global_model, client.criterion, client.global_optimizer,
                                                client.device, client.global_scheduler, single_batch=false)
            client.last_epoch += 1

        # Averaging models
        new_model = fednova_aggregation(old_model, [client.trained_model for client in network.clients], tau_i, weights,
                            network.clients[0].device)

        for client_idx in range(network.nb_clients):
            client = network.clients[client_idx]
            load_new_model(client.trained_model, new_model)
            assert equal(client.trained_model, network.clients[0].trained_model), \
                (f"Models 0 and {client.ID} are not equal.")
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
    """
        Computes the similarity ratio used to compute the collaboration weights.

        Parameters:
            x (float): Numerator (e.g., gradient difference).
            y (float): Denominator (e.g., reference gradient norm).

        Returns:
            float: Normalized difference ratio, clipped to [0, 1].
        """
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
    """
    Computes dynamic aggregation weights for a given client.

    Parameters:
        gradients (list[torch.Tensor]): Gradients from all clients.
        nb_points_by_clients (list[int]): Number of training samples per client.
        client_idx (int): Index of the client computing weights.
        numerators (list[list[float]]): Stores past gradient differences.
        denominators (list[float]): Stores past gradient norms.
        continuous (bool): If True, uses continuous weights; else, binary selection followed by renormalization.

    Returns:
        tuple:
            - list[float]: Normalized weights for each client.
            - numerators (updated)
            - denominators (updated)

    Notes:
        - Warns if a client does not collaborate with itself or if all weights are zero.
    """
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
    """
    Implementation of the "All-for-One" algorithm from our paper. .

    Parameters:
        network (Network): The federated network object containing all clients, models, and metadata.
        nb_of_synchronization (int): Number of communication rounds (global synchronizations).
        continuous (bool): If True, uses continuous weighting updates across rounds.
        pruning (bool): If True, checks whether the Optuna trial should be pruned.
        keep_track (bool): If True, stores model parameters and gradients after each local update step.

    Returns:
        Optional[Tuple[list, list]]: If `keep_track` is True, returns a tuple containing:
            - List of models tracked after each local step for each client.
            - List of gradients (scaled by learning rate) aggregated for each step and client.
    """
    try:
        # Estimate number of local updates per synchronization based on average dataset size.
        inner_iterations = int(np.mean([len(client.train_loader) for client in network.clients]))
    except TypeError:
        inner_iterations = 1

    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    # Compute Federated Averaging weights based on dataset sizes.
    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    fed_weights = [client.nb_train_points / total_nb_points for client in network.clients]

    # Evaluate initial performance on central server and log clients' metrics.
    loss_accuracy_central_server(network, fed_weights, network.writer, 0)
    for client in network.clients:
        client.write_train_val_test_performance()

    # Initialize gradient tracking numerators and denominators for weighting computation.
    numerators, denominators = [[[0] for _ in network.clients] for _ in network.clients], [[0] for _ in network.clients]

    if keep_track:
        # Store deep copies of model parameters and gradients at each step.
        track_models = [[[copy.deepcopy(m.data[0]).to("cpu") for m in c.trained_model.parameters()]] for c in network.clients]
        track_gradients = [[] for _ in network.clients]

    # Create an iterator over each client's training data for each model being evaluated.
    iter_loaders = [[iter(client.train_loader) for client in network.clients] for client in network.clients]

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"===============\tEpoch {synchronization_idx}\t===============")
        start_time = time.time()

        weights = {_: None for _ in range(network.nb_clients)}

        # Compute personalized weights for each client based on gradient similarity.
        for client_idx in range(network.nb_clients):
            gradients_eval = {_: None for _ in range(network.nb_clients)}
            client = network.clients[client_idx]

            # Evaluate gradients of client's model on each other client's data.
            for c_idx in range(network.nb_clients):
                c = network.clients[c_idx]
                gradient_eval, iter_loaders[client_idx][c_idx] = safe_gradient_computation(
                    c.train_loader, iter_loaders[client_idx][c_idx], c.device,
                    client.trained_model, client.criterion, client.optimizer, client.scheduler
                )
                gradients_eval[c_idx] = gradient_eval

            # Compute weights based on these evaluated gradients.
            weight, numerators, denominators = compute_weight_based_on_ratio(
                gradients_eval, network.nb_testpoints_by_clients,
                client_idx, numerators, denominators, continuous=continuous
            )

            weights[client_idx] = weight

            # Log histogram of computed weights.
            network.clients[client_idx].writer.add_histogram(
                'weights', np.array(weight),
                network.clients[client_idx].last_epoch * inner_iterations
            )

        for k in range(inner_iterations):

            # Compute the new model client by client.
            grad_time = time.time()

            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]
                gradients = []

                for c_idx in range(network.nb_clients):
                    c = network.clients[c_idx]
                    gradient, iter_loaders[client_idx][c_idx] = safe_gradient_computation(
                        c.train_loader, iter_loaders[client_idx][c_idx], c.device,
                        client.trained_model, client.criterion, client.optimizer, client.scheduler
                    )
                    gradients.append(gradient)

                # Aggregate the gradients using the computed weights.
                aggregated_gradients = aggregate_gradients(gradients, weights[client_idx], client.device)

                # Apply gradient update to the client's model.
                update_model(client.trained_model, aggregated_gradients, client.optimizer)

                # Optionally track model and gradient updates.
                if keep_track:
                    lr = client.optimizer.param_groups[0]['lr']
                    track_models[client_idx].append([copy.deepcopy(m.data[0]).to("cpu") for m in client.trained_model.parameters()])
                    track_gradients[client_idx].append([lr * g[0].to("cpu") for g in aggregated_gradients])

            print(f"Gradients computation time: {time.time() - grad_time} seconds")

        perf_time = time.time()

        # Evaluate clients' performance and update learning rates.
        for i in range(network.nb_clients):
            client = network.clients[i]
            client.last_epoch += 1
            client.write_train_val_test_performance()

        for client in network.clients:
            client.scheduler.step()

        # Evaluate performance on the central server.
        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)

        print(f"Performance time: {time.time() - perf_time} seconds")

        network.save()
        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")
        print_mem_usage()

        # Early stopping for hyperparameter search via Optuna.
        if pruning:
            if network.trial.should_prune():
                raise optuna.TrialPruned()

    # Cleanup GPU memory.
    torch.cuda.empty_cache()
    gc.collect()
    print_mem_usage("Memory usage at the end of the algo")

    if keep_track:
        return track_models, track_gradients

def all_for_all_algo(network: Network, nb_of_synchronization: int = 5, pruning: bool = False,
                     keep_track=False, collab_based_on="ratio"):
    """
    Implementation of the "All-for-All" algorithm from Even et al using our collaboration criterion.

    Parameters:
        network (Network): The federated network object containing all clients, models, and metadata.
        nb_of_synchronization (int): Number of communication rounds (global synchronizations).
        pruning (bool): If True, checks whether the Optuna trial should be pruned.
        keep_track (bool): If True, stores model parameters and gradients after each local update step.
        collab_based_on (str): Strategy for collaboration, either "local" (self-only) or "ratio" (cross-client).

    Returns:
        Optional[Tuple[list, list]]: If `keep_track` is True, returns a tuple containing:
            - List of models tracked after each local step for each client.
            - List of gradients (scaled by learning rate) aggregated for each step and client.
    """

    # Compute average number of local iterations per synchronization round
    try:
        inner_iterations = int(np.mean([len(client.train_loader) for client in network.clients]))
    except TypeError:
        inner_iterations = 1
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    # Compute normalized weights for weighted global evaluation
    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    fed_weights = [client.nb_train_points / total_nb_points for client in network.clients]

    # Evaluate initial global performance (before training)
    loss_accuracy_central_server(network, fed_weights, network.writer, 0)
    for client in network.clients:
        client.write_train_val_test_performance()

    # Initialize storage for weights computation
    numerators, denominators = [[[] for _ in network.clients] for _ in network.clients], [[] for _ in network.clients]

    # Optionally track model states and gradients
    if keep_track:
        track_models = [[[m.data[0].to("cpu") for m in c.trained_model.parameters()]] for c in network.clients]
        track_gradients = [[] for _ in network.clients]

    # Create data loader iterators for each client
    iter_loaders = [iter(client.train_loader) for client in network.clients]

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"===============\tEpoch {synchronization_idx}\t===============")
        start_time = time.time()

        weights = {_: None for _ in range(network.nb_clients)}

        if collab_based_on == "local":
            # Each client uses only its own gradients (no collaboration)
            for client_idx in range(network.nb_clients):
                weight = [int(j == client_idx) for j in range(network.nb_clients)]
                weights[client_idx] = weight
                network.clients[client_idx].writer.add_histogram(
                    'weights', np.array(weight),
                    network.clients[client_idx].last_epoch * inner_iterations
                )

        elif collab_based_on == "ratio":
            gradients_eval = {_: None for _ in range(network.nb_clients)}
            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]
                gradient_eval, iter_loaders[client_idx] = safe_gradient_computation(
                    client.train_loader, iter_loaders[client_idx], client.device,
                    client.trained_model, client.criterion, client.optimizer,
                    client.scheduler
                )
                gradients_eval[client_idx] = gradient_eval

            for client_idx in range(network.nb_clients):
                weight, numerators, denominators = compute_weight_based_on_ratio(
                    gradients_eval,
                    network.nb_testpoints_by_clients,
                    client_idx, numerators, denominators,
                    False  # continuous=False
                )
                weights[client_idx] = weight
                network.clients[client_idx].writer.add_histogram(
                    'weights', np.array(weight),
                    network.clients[client_idx].last_epoch * inner_iterations
                )
        else:
            raise ValueError("Collaboration criterion '{0}' is not recognized.".format(collab_based_on))

        for k in range(inner_iterations):
            gradients = []

            # Compute local gradients for each client
            grad_time = time.time()
            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]
                gradient, iter_loaders[client_idx] = safe_gradient_computation(
                    client.train_loader, iter_loaders[client_idx], client.device,
                    client.trained_model, client.criterion, client.optimizer,
                    client.scheduler
                )
                gradients.append(gradient)
            print(f"Gradients computation time: {time.time() - grad_time} seconds")

            # Update models using aggregated gradients (weighted combination)
            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]

                aggregated_gradients = aggregate_gradients(gradients, weights[client_idx], client.device)
                update_model(client.trained_model, aggregated_gradients, client.optimizer)

                # Optionally store model and gradient after update
                if keep_track:
                    lr = client.optimizer.param_groups[0]['lr']
                    track_models[client_idx].append([m.data[0].to("cpu") for m in client.trained_model.parameters()])
                    track_gradients[client_idx].append([lr * g[0].to("cpu") for g in aggregated_gradients])

        perf_time = time.time()
        for client in network.clients:
            client.last_epoch += 1
            client.write_train_val_test_performance()
            client.scheduler.step()

        # Evaluate global performance
        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)

        print(f"Performance time: {time.time() - perf_time} seconds")
        network.save()
        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")
        print_mem_usage()

        # Handle pruning logic if Optuna trial is active
        if pruning and network.trial.should_prune():
            raise optuna.TrialPruned()

    # Clean up GPU and memory
    torch.cuda.empty_cache()
    gc.collect()
    print_mem_usage("Memory usage at the end of the algo")

    # Return tracked model and gradient history if enabled
    if keep_track:
        return track_models, track_gradients


def cobo_algo(network: Network, nb_of_synchronization: int = 5, pruning: bool = False, keep_track=False):
    """https://proceedings.neurips.cc/paper_files/paper/2024/file/1c32452f112719f7c1db6d983d060f78-Paper-Conference.pdf"""

    # Compute average number of local iterations per synchronization round
    try:
        inner_iterations = int(np.mean([len(client.train_loader) for client in network.clients]))
    except TypeError:
        inner_iterations = 1
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    # Compute normalized weights for weighted global evaluation
    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    fed_weights = [client.nb_train_points / total_nb_points for client in network.clients]

    # Evaluate initial global performance (before training)
    loss_accuracy_central_server(network, fed_weights, network.writer, 0)
    for client in network.clients:
        client.write_train_val_test_performance()

    # Optionally track model states and gradients
    if keep_track:
        track_models = [[[m.data[0].to("cpu") for m in c.trained_model.parameters()]] for c in network.clients]
        track_gradients = [[] for _ in network.clients]

    # Create an iterator over each client's training data for each model being evaluated.
    iter_loaders = [[iter(client.train_loader) for client in network.clients] for client in network.clients]

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"===============\tEpoch {synchronization_idx}\t===============")
        start_time = time.time()

        weights = {_: [1 for e in range(network.nb_clients)] for _ in range(network.nb_clients)}

        # Compute personalized weights for each client based on gradient similarity.
        for client_idx in range(network.nb_clients):
            # gradients_eval = {_: None for _ in range(network.nb_clients)}
            client = network.clients[client_idx]

            # Evaluate gradients of client's model on each other client's data.
            for c_idx in range(network.nb_clients):
                if c_idx == client_idx:
                    continue
                c = network.clients[c_idx]

                average_model = aggregate_models([client.trained_model, c.trained_model],
                                                 [0.5, 0.5], client.device)
                optimizer = optim.SGD(average_model.parameters(), lr=0,
                                      momentum=0, weight_decay=0)
                gradient_i, iter_loaders[client_idx][client_idx] = safe_gradient_computation(
                    client.train_loader, iter_loaders[client_idx][client_idx], client.device,
                    average_model, client.criterion, optimizer, client.scheduler
                )
                gradient_k, iter_loaders[client_idx][c_idx] = safe_gradient_computation(
                    c.train_loader, iter_loaders[client_idx][c_idx], c.device,
                    average_model, c.criterion, optimizer, c.scheduler
                )

                # Choice of gamma as in the original repository.
                # See https://github.com/epfml/CoBo/blob/main/personalized-vision-models/grouping.py#L187
                gamma = 0.01
                g_i = torch.concat([p.flatten() for p in gradient_i if p is not None])
                g_k = torch.concat([p.flatten() for p in gradient_i if p is not None])
                weights[client_idx][c_idx] = torch.clamp(weights[client_idx][c_idx] + gamma * g_i.T @ g_k, 0, 1)

        for k in range(inner_iterations):
            # Compute the new model client by client.
            grad_time = time.time()
            # Save models at previous iteration
            models = [_.trained_model for _ in network.clients]

            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]
                # We need now only N iterators.
                gradient, iter_loaders[client_idx][client_idx] = safe_gradient_computation(
                    client.train_loader, iter_loaders[client_idx][client_idx], client.device,
                    client.trained_model, client.criterion, client.optimizer, client.scheduler
                )

                client = network.clients[client_idx]

                aggregated_models = aggregate_models(
                    [aggregate_models([models[client_idx], m], [1, -1], client.device) for m in models],
                    weights, client.device)

                # Aggregate the gradients using the computed weights.
                rho = 1 / client.step_size
                momentum = [p.data for p in scalar_multiplication(aggregated_models, rho).parameters()]
                aggregated_gradients = gradient + momentum

                # Apply gradient update to the client's model.
                update_model(client.trained_model, aggregated_gradients, client.optimizer)

                # Optionally track model and gradient updates.
                if keep_track:
                    lr = client.optimizer.param_groups[0]['lr']
                    track_models[client_idx].append(
                        [copy.deepcopy(m.data[0]).to("cpu") for m in client.trained_model.parameters()])
                    track_gradients[client_idx].append([lr * g[0].to("cpu") for g in aggregated_gradients])

            print(f"Gradients computation time: {time.time() - grad_time} seconds")

        perf_time = time.time()

        # Evaluate clients' performance and update learning rates.
        for i in range(network.nb_clients):
            client = network.clients[i]
            client.last_epoch += 1
            client.write_train_val_test_performance()

        for client in network.clients:
            client.scheduler.step()

        # Evaluate performance on the central server.
        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)

        print(f"Performance time: {time.time() - perf_time} seconds")

        network.save()
        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")
        print_mem_usage()

        # Early stopping for hyperparameter search via Optuna.
        if pruning:
            if network.trial.should_prune():
                raise optuna.TrialPruned()

        # Cleanup GPU memory.
    torch.cuda.empty_cache()
    gc.collect()
    print_mem_usage("Memory usage at the end of the algo")

    if keep_track:
        return track_models, track_gradients

def ditto_algo(network: Network, nb_of_synchronization: int = 5, pruning: bool = False, keep_track=False):

    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    fed_weights = [client.nb_train_points / total_nb_points for client in network.clients]

    try:
        inner_iterations = int(np.mean([len(client.train_loader) for client in network.clients]))
    except TypeError:
        inner_iterations = 1
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    loss_accuracy_central_server(network, fed_weights, network.writer, 0)
    for client in network.clients:
        assert equal(client.trained_model, network.clients[0].trained_model), \
            (f"Models 0 and {client.ID} are not equal.")
        client.write_train_val_test_performance()


    if keep_track:
        track_models = [[[m.data[0].to("cpu") for m in c.trained_model.parameters()]] for c in network.clients]
        # track_gradients = [[] for c in network.clients]

    # We create a data iterator for each clients, for each trained model by clients.
    iter_loaders = [iter(client.train_loader) for client in network.clients]

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"=============== \tEpoch {synchronization_idx} ===============")
        start_time = time.time()

        ################### First training the global model. ####################
        # One pass of local training
        for i in range(network.nb_clients):
            client = network.clients[i]
            iter_loaders[i] = continue_training(inner_iterations, client.train_loader, iter_loaders[i],
                                                client.global_model, client.criterion, client.global_optimizer,
                                                client.device, client.global_scheduler)
            client.last_epoch += inner_iterations

        # Averaging models
        new_model = aggregate_models([client.global_model for client in network.clients],
                                     [1 / network.nb_clients for _ in range(network.nb_clients)],
                                     network.clients[0].device)
        ########################################

        #################### Train personalized model ####################
        # One pass of local training
        for client_idx in range(network.nb_clients):
            client = network.clients[client_idx]
            for k in range(inner_iterations):
                gradient, iter_loaders[client_idx] = safe_gradient_computation(
                    client.train_loader, iter_loaders[client_idx], client.device,
                    client.trained_model, client.criterion, client.optimizer,
                    client.scheduler
                )

                # print("GRADIENT")
                # print(gradient)
                #
                # print("TRAINED MODEL")
                # print([_ for _ in client.trained_model.parameters()])
                #
                # print("PERSONALIZED MODEL")
                # print([_ for _ in client.global_model.parameters()])
                #
                # for param_group in client.global_optimizer.param_groups:
                #     for param in param_group['params']:
                #         print(param)
                #         print(param.grad)

                lbda = 1
                # # Using global_model at previous iteration.
                reg = aggregate_models([client.trained_model, client.global_model], [lbda, -lbda],
                                       client.device)

                aggregated_gradients = gradient + [p.data for p in reg.parameters()]
                # Apply gradient update to the client's model.
                update_model(client.trained_model, aggregated_gradients, client.optimizer)
        ########################################

        #################### Update the global model ####################
        for client_idx in range(network.nb_clients):
            client = network.clients[client_idx]
            load_new_model(client.global_model, new_model)
            assert equal(client.global_model, network.clients[0].global_model), \
                (f"Models 0 and {client.ID} are not equal.")
        ########################################

        perf_time = time.time()
        for client in network.clients:
            client.last_epoch += 1
            client.write_train_val_test_performance()
            client.scheduler.step()

        # Evaluate global performance
        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)

        print(f"Performance time: {time.time() - perf_time} seconds")
        network.save()
        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")
        print_mem_usage()

        # Handle pruning logic if Optuna trial is active
        if pruning and network.trial.should_prune():
            raise optuna.TrialPruned()

        # Clean up GPU and memory
    torch.cuda.empty_cache()
    gc.collect()
    print_mem_usage("Memory usage at the end of the algo")

    # Return tracked model and gradient history if enabled
    if keep_track:
        return track_models  # , track_gradients
    return None

def waffle_algo(network: Network, nb_of_synchronization: int = 5, pruning: bool = False, keep_track=False):
    # Compute average number of local iterations per synchronization round
    try:
        inner_iterations = int(np.mean([len(client.train_loader) for client in network.clients]))
    except TypeError:
        inner_iterations = 1
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    # Compute normalized weights for weighted global evaluation
    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    fed_weights = [client.nb_train_points / total_nb_points for client in network.clients]

    # Evaluate initial global performance (before training)
    loss_accuracy_central_server(network, fed_weights, network.writer, 0)
    for client in network.clients:
        client.write_train_val_test_performance()

    # Initialize storage for weights computation
    numerators, denominators = [[[] for _ in network.clients] for _ in network.clients], [[] for _ in
                                                                                          network.clients]

    # Optionally track model states and gradients
    if keep_track:
        track_models = [[[m.data[0].to("cpu") for m in c.trained_model.parameters()]] for c in network.clients]
        track_gradients = [[] for _ in network.clients]

    # Create data loader iterators for each client
    iter_loaders = [iter(client.train_loader) for client in network.clients]

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"===============\tEpoch {synchronization_idx}\t===============")
        start_time = time.time()

        weights = {_: None for _ in range(network.nb_clients)}
        perf_time = time.time()

        # Evaluate clients' performance and update learning rates.
        for i in range(network.nb_clients):
            client = network.clients[i]
            client.last_epoch += 1
            client.write_train_val_test_performance()

        for client in network.clients:
            client.scheduler.step()

        # Evaluate performance on the central server.
        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)

        print(f"Performance time: {time.time() - perf_time} seconds")

        network.save()
        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")
        print_mem_usage()

        # Early stopping for hyperparameter search via Optuna.
        if pruning:
            if network.trial.should_prune():
                raise optuna.TrialPruned()
    # Clean up GPU and memory
    torch.cuda.empty_cache()
    gc.collect()
    print_mem_usage("Memory usage at the end of the algo")
