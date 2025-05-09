import copy
import time
from random import sample

import numpy as np
import optuna
import torch
from tqdm import tqdm

from src.data.Network import Network
from src.optim.PytorchUtilities import print_collaborating_clients, aggregate_models, \
    get_models_of_collaborating_models, equal, load_new_model, aggregate_gradients, fednova_aggregation
from src.optim.Train import compute_loss_and_accuracy, update_model, gradient_step, write_grad, compute_gradient_validation_set, safe_gradient_computation
from src.plot.PlotDistance import plot_pvalues
from src.quantif.Distances import compute_matrix_of_distances, acceptance_pvalue, \
    rejection_pvalue
from src.quantif.Metrics import Metrics

def ratio(x,y):
    if x == 0:
        return 1
    if y == 0:
        return 0
    else:
        assert 1 - x/y < 1, "Not lower than one."
        return max(1 - x/y, 0)

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
        old_model = deepcopy(network.clients[0].trained_model)
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
        print(f"Elapsed time: {time.time() - start_time} seconds")
        network.save()
    if keep_track:
        return track_models
    return None

def compute_weight_based_on_gradient(gradients, validation_gradient, norm_validation_gradient, client_idx):
    grad_norm_diff = []
    for _ in range(len(gradients)):
        grads_G = [p.flatten() for p in gradients[_] if p is not None]
        grad_norm_diff.append(
            torch.sum(torch.cat([(gF - gG).pow(2) for gF, gG in zip(grads_G, validation_gradient)]))
        )

    weight = [float(d < norm_validation_gradient) for d in grad_norm_diff]
    if sum(weight) == 0:
        return [int(i == client_idx) for i in range(len(gradients))]

    total_weight = sum(weight)
    return [w / total_weight for w in weight]

def compute_weight_based_on_ratio(gradients, validation_gradients, client_idx, numerators, denominators,
                                  continuous: bool = False) :

    # grads, grads_val = [], []
    # for _ in range(len(gradients)):
    #     grads.append(torch.concat([p.flatten() for p in gradients[_] if p is not None]))
    #     grads_val.append(torch.concat([p.flatten() for p in validation_gradients[_] if p is not None]))

    new_denom = gradients[client_idx].T @ gradients[client_idx] # grads[client_idx] @ grads_val[client_idx]
    denominators[client_idx].append(new_denom.item())

    for _ in range(len(gradients)):
        new_num = torch.linalg.vector_norm(gradients[client_idx] - gradients[_])
        numerators[client_idx][_].append(new_num.item())

    if True:
        # Réfléchir un peu plus!
        weight = [ratio(numerators[client_idx][_][-1], denominators[client_idx][-1]) for _ in range(len(gradients))]
        if sum(weight) == 0:
            print(f"⚠️ The sum of weights is zero.")
            weight = [int(i == client_idx) for i in range(len(gradients))]
        total_weight = sum(weight)
        return [w / total_weight for w in weight], numerators, denominators
    weight = [1 if numerators[client_idx][_][-1] == 0 else int(1 - numerators[client_idx][_][-1] / denominators[client_idx][-1] > 0) for _ in range(len(gradients))]
    if sum(weight) == 0:
        print(f"⚠️ The sum of weights is zero.")
        weight = [int(i == client_idx) for i in range(len(gradients))]
    total_weight = sum(weight)
    return [w / total_weight for w in weight], numerators, denominators

def compute_weight_based_on_scalar_product(gradients, validation_gradient, norm_validation_gradient, client_idx):
    sclpdts = []
    validation_gradient = torch.cat([p.flatten() for p in validation_gradient if p is not None])
    for _ in range(len(gradients)):
        grads_G = torch.cat([p.flatten() for p in gradients[_] if p is not None])
        sclpdts.append(
            torch.dot(grads_G, validation_gradient)
        )

    weight = [float(s <= sclpdts[client_idx]) for s in sclpdts]
    if sum(weight) == 0:
        raise ValueError("The sum of weights should not be equal to zero.")

    total_weight = sum(weight)
    return [w / total_weight for w in weight]

def all_for_one_algo(network: Network, nb_of_synchronization: int = 5, inner_iterations: int = 50,
                     plot_matrix: bool = True, pruning: bool = False, logs="light",
                     collab_based_on="grad", keep_track=False):
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

    metrics_for_comparison = Metrics(f"{network.dataset_name}_{network.algo_name}",
                              "_comparaison", network.nb_clients, network.nb_testpoints_by_clients)

    numerators, denominators = [[[] for _ in network.clients] for _ in network.clients], [[] for _ in network.clients]
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
            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]

                gradients = []
                gradients2 = []
                gradients3 = []
                norm_validation_gradients = []

                # Computing gradients on each clients.
                grad_time = time.time()
                for c_idx in range(network.nb_clients):
                    c = network.clients[c_idx]
                    # set_seed(client.last_epoch * inner_iterations + k)

                    gradient = safe_gradient_computation(c.train_loader, iter_loaders[client_idx][c_idx], c.device,
                                                           client.trained_model, client.criterion, client.optimizer,
                                                           client.scheduler)
                    gradients.append(gradient)

                    # if collab_based_on in ["ratio", "pdtscl", "grad"]:
                    gradient2 = c.train_loader.dataset.covariance @ (client.trained_model.linear.weight.data
                                                                     - c.train_loader.dataset.true_theta).T  #safe_gradient_computation(c.train_loader, iter_loaders[client_idx][c_idx], c.device,
                                                           # client.trained_model, client.criterion, client.optimizer,
                                                           # client.scheduler)
                    gradients2.append(gradient2)
                    gradient3 = safe_gradient_computation(c.train_loader, iter_loaders[client_idx][c_idx], c.device,
                                                          client.trained_model, client.criterion, client.optimizer,
                                                          client.scheduler)
                    gradients3.append(gradient3)

                    if collab_based_on in ["grad", "pdtscl"]:
                        norm_validation_gradients.append(
                            torch.sum(torch.cat([gF.pow(2) for gF in [p.flatten() for p in gradient2]])))

                print(f"Gradients computation time: {time.time() - grad_time} seconds")
                weight, numerators, denominators = compute_weight_based_on_ratio(gradients2, gradients3,
                                                                                 client_idx, numerators,
                                                                                 denominators, False)
                client = network.clients[client_idx]
                if collab_based_on == "grad":
                    weight = compute_weight_based_on_gradient(gradients, [p.flatten() for p in gradients2[client_idx]],
                                                           norm_validation_gradients[client_idx], client_idx)
                elif collab_based_on == "loss":
                    metrics_for_comparison.reinitialize()
                    compute_matrix_of_distances(rejection_pvalue, network, metrics_for_comparison, symetric_distance=True)
                    weight = [int(client_idx == j or metrics_for_comparison.aggreage_heter()[client_idx][j] > 0.05) for j in range(network.nb_clients)]

                    weight = [w / sum(weight) for w in weight]
                elif collab_based_on == "pdtscl":
                    weight = compute_weight_based_on_scalar_product(gradients, [p.flatten() for p in gradients2[client_idx]],
                                                              norm_validation_gradients[client_idx], client_idx)
                elif collab_based_on == "local":
                    weight = [int(j == client_idx) for j in range(network.nb_clients)]
                elif collab_based_on == "ratio":
                    pass
                    # weight, numerators, denominators = compute_weight_based_on_ratio(gradients2, gradients3,
                    #                                                                  client_idx, numerators,
                    #                                                                  denominators, False)
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

        for i in range(network.nb_clients):
            client = network.clients[i]
            client.last_epoch += 1
            client.write_train_val_test_performance()

        for client in network.clients:
            client.scheduler.step()

        print(f"Elapsed time: {time.time() - start_time} seconds")
        print("Now computing matrix of distances...")
        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)
        network.save()

        # The network has trial parameter only if the pruning is active (for hyperparameters search).
        if pruning:
            if network.trial.should_prune():
                raise optuna.TrialPruned()

        if synchronization_idx % 20 == 1:
            ### We compute the distance between clients.
            metrics_for_comparison.reinitialize()
            metrics_for_comparison.add_distances(weights)
            plot_pvalues(metrics_for_comparison, f"{synchronization_idx}")

    if keep_track:
        return track_models, track_gradients

def all_for_all_algo(network: Network, nb_of_synchronization: int = 5, inner_iterations: int = 50,
                     plot_matrix: bool = True, pruning: bool = False, logs="light",
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

    metrics_for_comparison = Metrics(f"{network.dataset_name}_{network.algo_name}",
                                     "_comparaison", network.nb_clients, network.nb_testpoints_by_clients)


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
            norm_validation_gradients = []

            # Computing gradients on each clients.
            grad_time = time.time()
            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]
                # set_seed(client.last_epoch * inner_iterations + k)

                gradient = safe_gradient_computation(client.train_loader, iter_loaders[client_idx], client.device,
                                                     client.trained_model, client.criterion, client.optimizer,
                                                     client.scheduler)

                gradients.append(gradient)

                # if collab_based_on in ["ratio", "pdtscl", "grad"]:
                gradient2 = client.train_loader.dataset.covariance @ (client.trained_model.linear.weight.data
                                                                 - client.train_loader.dataset.true_theta).T
                # gradient2 = safe_gradient_computation(client.train_loader, iter_loaders[client_idx], client.device,
                #                                  client.trained_model, client.criterion, client.optimizer,
                #                                  client.scheduler)
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
                weight, numerators, denominators = compute_weight_based_on_ratio(gradients2, gradients3,
                                                                            client_idx, numerators,
                                                                            denominators, False)
                client = network.clients[client_idx]
                if collab_based_on in ["grad", "pdtscl"]:
                    norm_validation_gradients.append(
                        torch.sum(torch.cat([gF.pow(2) for gF in [p.flatten() for p in gradient2]])))

                if collab_based_on in ["grad", "pdtscl"]:
                    norm_validation_gradients.append(
                        torch.sum(torch.cat([gF.pow(2) for gF in [p.flatten() for p in gradient2]])))

                if collab_based_on == "grad":
                    weight = compute_weight_based_on_gradient(gradients, [p.flatten() for p in gradients2[client_idx]],
                                                              norm_validation_gradients[client_idx], client_idx)
                elif collab_based_on == "loss":
                    metrics_for_comparison.reinitialize()
                    compute_matrix_of_distances(rejection_pvalue, network, metrics_for_comparison,
                                                symetric_distance=True)
                    weight = [int(client_idx == j or metrics_for_comparison.aggreage_heter()[client_idx][j] > 0.05) for
                              j in range(network.nb_clients)]

                    weight = [w / sum(weight) for w in weight]
                elif collab_based_on == "pdtscl":
                    weight = compute_weight_based_on_scalar_product(gradients,
                                                                    [p.flatten() for p in gradients2[client_idx]],
                                                                    norm_validation_gradients[client_idx], client_idx)
                elif collab_based_on == "local":
                    weight = [int(j == client_idx) for j in range(network.nb_clients)]
                elif collab_based_on == "ratio":
                    pass
                #     weight, numerators, denominators = compute_weight_based_on_ratio(gradients, gradients2,
                #                                                                      client_idx, numerators,
                #                                                                      denominators, 0.5)
                else:
                    raise ValueError("Collaboration criterion '{0}' is not recognized.".format(collab_based_on))
                weights.append(weight)
                client.writer.add_histogram('weights', np.array(weight), client.last_epoch * inner_iterations + k)

            nb_collaborations = [nb_collaborations[i] + np.sum(weights[i] != 0) for i in range(network.nb_clients)]

            for client_idx in range(network.nb_clients):
                client = network.clients[client_idx]
                # We plot the computed gradient on each client before their aggregation.
                if logs=="full":
                    write_grad(client.trained_model, client.writer, client.writer)

                aggregated_gradients = aggregate_gradients(gradients, weights[client_idx], client.device)
                update_model(client.trained_model, aggregated_gradients, client.optimizer)

                if keep_track:
                    lr = client.optimizer.param_groups[0]['lr']
                    track_models[client_idx].append([m.data[0].to("cpu") for m in client.trained_model.parameters()])
                    track_gradients[client_idx].append([lr * g[0].to("cpu") for g in aggregated_gradients])
        print("Step-size:", client.optimizer.param_groups[0]['lr'])
        for client in network.clients:
            client.last_epoch += 1
            client.write_train_val_test_performance()
        for client in network.clients:
            client.scheduler.step()

        print(f"Elapsed time: {time.time() - start_time} seconds")
        loss_accuracy_central_server(network, fed_weights, network.writer, client.last_epoch)
        network.save()

        # The network has trial parameter only if the pruning is active (for hyperparameters search).
        if pruning:
            if network.trial.should_prune():
                raise optuna.TrialPruned()

        if synchronization_idx % 20 == 1:
            ### We compute the distance between clients.
            metrics_for_comparison.reinitialize()
            metrics_for_comparison.add_distances(weights)
            plot_pvalues(metrics_for_comparison, f"{synchronization_idx}")

    if keep_track:
        return track_models, track_gradients


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

    total_nb_points = np.sum([client.nb_train_points for client in network.clients])
    weights = [client.nb_train_points / total_nb_points for client in network.clients]

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

