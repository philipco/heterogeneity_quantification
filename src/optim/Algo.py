import copy
import time

import numpy as np
import optuna
import torch

from src.data.Network import Network
from src.optim.PytorchUtilities import aggregate_models, equal, load_new_model, aggregate_gradients, fednova_aggregation
from src.optim.Train import compute_loss_and_accuracy, update_model, gradient_step, write_grad, compute_gradient_validation_set, safe_gradient_computation
from src.plot.PlotDistance import plot_pvalues
from src.quantif.Metrics import Metrics

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


def federated_training(network: Network, nb_of_synchronization: int = 5, nb_of_local_epoch: int = 1, keep_track: bool = False):

    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    weights = [client.nb_train_points / total_nb_points for client in network.clients]

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

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"=============== \tEpoch {synchronization_idx} ===============")
        start_time = time.time()
        # One pass of local training
        for i in range(network.nb_clients):
            network.clients[i].continue_training(nb_of_local_epoch, synchronization_idx, single_batch=False)

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
        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")
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
    grads, grads_val = [], []
    for _ in range(len(gradients)):
        grads.append(torch.concat([p.flatten() for p in gradients[_] if p is not None]))
        # grads.append(gradients[_])

    new_denom = grads[client_idx].T @ grads[client_idx] # grads[client_idx] @ grads_val[client_idx]
    # new_denom = grads[client_idx].T @ grads_val[client_idx]
    denominators[client_idx].append(new_denom.item())

    for _ in range(len(gradients)):
        new_num = torch.linalg.vector_norm(grads[client_idx] - grads[_])
        # new_num = (grads[client_idx] @ grads[_] + grads_val[client_idx] @ grads_val[_]
        #            - grads[_] @ grads_val[_])
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

def all_for_one_algo(network: Network, nb_of_synchronization: int = 5, pruning: bool = False,
                     collab_based_on="grad", keep_track=False):
    try:
        inner_iterations = 1 #int(np.mean([len(client.train_loader) for client in network.clients]))
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

        for k in range(inner_iterations):
            weights = []

            # Compute the new model client by client.
            grad_time = time.time()
            gradients2 = {_: None for _ in range(network.nb_clients)}
            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]

                gradients = []
                gradients3 = []

                for c_idx in range(network.nb_clients):
                    c = network.clients[c_idx]

                    gradient = safe_gradient_computation(c.train_loader, iter_loaders[client_idx][c_idx], c.device,
                                                           client.trained_model, client.criterion, client.optimizer,
                                                           client.scheduler)
                    gradients.append(gradient)

                    gradient2 = safe_gradient_computation(c.train_loader, iter_loaders[client_idx][c_idx], c.device,
                                                          client.trained_model, client.criterion, client.optimizer,
                                                          client.scheduler)
                    if gradients2[c_idx] is None:
                        gradients2[c_idx] = gradient2
                    else:
                        beta = 0. #1-1 / synchronization_idx
                        gradients2[c_idx] = [(g_old * beta + (1-beta)* g_new) if g_new is not None else None for (g_old, g_new) in zip(gradients2[c_idx], gradient2)]
                        # gradients2[c_idx] = gradients2[c_idx] / (1-beta)
                    # gradients2[c_idx] = c.train_loader.dataset.covariance @ (client.trained_model.linear.weight.data.to("cpu")
                    #                                                  - c.train_loader.dataset.true_theta).T
                    # gradients2.append(gradient2)
                    gradient3 = safe_gradient_computation(c.train_loader, iter_loaders[client_idx][c_idx], c.device,
                                                          client.trained_model, client.criterion, client.optimizer,
                                                          client.scheduler)
                    gradients3.append(gradient3)


                weight, numerators, denominators = compute_weight_based_on_ratio(gradients2, network.nb_testpoints_by_clients,
                                                                                 client_idx, numerators,
                                                                                 denominators, False)
                client = network.clients[client_idx]
                if collab_based_on == "local":
                    weight = [int(j == client_idx) for j in range(network.nb_clients)]
                elif collab_based_on == "ratio":
                    pass
                else:
                    raise ValueError("Collaboration criterion '{0}' is not recognized.".format(collab_based_on))

                weights.append(weight)

                aggregated_gradients = aggregate_gradients(gradients, weights[client_idx], client.device)
                update_model(client.trained_model, aggregated_gradients, client.optimizer)
                client.writer.add_histogram('weights', np.array(weight), client.last_epoch * inner_iterations + k)
                if keep_track:
                    lr = client.optimizer.param_groups[0]['lr']
                    track_models[client_idx].append([copy.deepcopy(m.data[0]).to("cpu") for m in client.trained_model.parameters()])
                    track_gradients[client_idx].append([lr * g[0].to("cpu") for g in aggregated_gradients])
            print(f"Gradients computation time: {time.time() - grad_time} seconds")
        for i in range(network.nb_clients):
            client = network.clients[i]
            client.last_epoch += 1
            client.write_train_val_test_performance()

        for client in network.clients:
            client.scheduler.step()

        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)
        network.save()
        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")

        # The network has trial parameter only if the pruning is active (for hyperparameters search).
        if pruning:
            if network.trial.should_prune():
                raise optuna.TrialPruned()

    if keep_track:
        return track_models, track_gradients

def all_for_all_algo(network: Network, nb_of_synchronization: int = 5, pruning: bool = False,
                     keep_track=False, collab_based_on="loss"):
    try:
        inner_iterations = 1 #int(np.mean([len(client.train_loader) for client in network.clients]))
    except TypeError:
        inner_iterations = 1
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")

    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    fed_weights = [client.nb_train_points / total_nb_points for client in network.clients]

    loss_accuracy_central_server(network, fed_weights, network.writer, network.nb_initial_epochs)
    for client in network.clients:
        client.write_train_val_test_performance()

    numerators, denominators = [[[] for _ in network.clients] for _ in network.clients], [[] for _ in network.clients]

    nb_collaborations = [0 for client in network.clients]

    if keep_track:
        track_models = [[[m.data[0].to("cpu") for m in c.trained_model.parameters()]] for c in network.clients]
        track_gradients = [[] for c in network.clients]

    iter_loaders = [iter(client.train_loader) for client in network.clients]

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"===============\tEpoch {synchronization_idx}\t===============")
        start_time = time.time()

        for k in range(inner_iterations):

            weights = []
            gradients = []
            gradients2 = []
            gradients3 = []

            # Computing gradients on each clients.
            grad_time = time.time()
            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]

                gradient = safe_gradient_computation(client.train_loader, iter_loaders[client_idx], client.device,
                                                     client.trained_model, client.criterion, client.optimizer,
                                                     client.scheduler)

                gradients.append(gradient)

                # gradient2 = client.train_loader.dataset.covariance @ (client.trained_model.linear.weight.data.to("cpu")
                #                                                  - client.train_loader.dataset.true_theta).T
                gradient2 = safe_gradient_computation(client.train_loader, iter_loaders[client_idx], client.device,
                                                 client.trained_model, client.criterion, client.optimizer,
                                                 client.scheduler)
                gradients2.append(gradient2)
                gradient3 = safe_gradient_computation(client.train_loader, iter_loaders[client_idx], client.device,
                                                      client.trained_model, client.criterion, client.optimizer,
                                                      client.scheduler)
                gradients3.append(gradient3)

            print(f"Gradients computation time: {time.time() - grad_time} seconds")

            # Computing the weights for each
            for client_idx in range(network.nb_clients):

                # Presently, we always compute the ratio weigth (and override it) it requires.
                # In order to plot the ratio for every algo.
                weight, numerators, denominators = compute_weight_based_on_ratio(gradients2, network.nb_testpoints_by_clients,
                                                                            client_idx, numerators,
                                                                            denominators, False)
                client = network.clients[client_idx]

                if collab_based_on == "local":
                    weight = [int(j == client_idx) for j in range(network.nb_clients)]
                elif collab_based_on == "ratio":
                    pass
                else:
                    raise ValueError("Collaboration criterion '{0}' is not recognized.".format(collab_based_on))
                weights.append(weight)
                client.writer.add_histogram('weights', np.array(weight), client.last_epoch * inner_iterations + k)

            nb_collaborations = [nb_collaborations[i] + np.sum(weights[i] != 0) for i in range(network.nb_clients)]

            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]

                aggregated_gradients = aggregate_gradients(gradients, weights[client_idx], client.device)
                update_model(client.trained_model, aggregated_gradients, client.optimizer)

                if keep_track:
                    lr = client.optimizer.param_groups[0]['lr']
                    track_models[client_idx].append([m.data[0].to("cpu") for m in client.trained_model.parameters()])
                    track_gradients[client_idx].append([lr * g[0].to("cpu") for g in aggregated_gradients])
        for client in network.clients:
            client.last_epoch += 1
            client.write_train_val_test_performance()
        for client in network.clients:
            client.scheduler.step()

        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)
        network.save()

        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        print(f"Elapsed time: {time.time() - start_time} seconds")

        # The network has trial parameter only if the pruning is active (for hyperparameters search).
        if pruning:
            if network.trial.should_prune():
                raise optuna.TrialPruned()

    if keep_track:
        return track_models, track_gradients

