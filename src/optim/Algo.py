import copy
from random import sample

import numpy as np
from sklearn.preprocessing import normalize
from torch.utils.tensorboard import SummaryWriter

from src.data.Network import Network
from src.optim.PytorchUtilities import print_collaborating_clients, aggregate_models, \
    get_models_of_collaborating_models, equal, load_new_model, aggregate_gradients
from src.optim.Train import compute_loss_and_accuracy, update_model, gradient_step, \
    write_train_val_test_performance
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
    writer.close()


def federated_training(network: Network, nb_of_synchronization: int = 5, nb_of_local_epoch: int = 1):
    writer = SummaryWriter(
        log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/test/{network.dataset_name}_{network.algo_name}_'
                f'central_server')

    total_nb_points = np.sum([len(client.train_loader.dataset) for client in network.clients])
    weights = [len(client.train_loader.dataset) / total_nb_points for client in network.clients]
    loss_accuracy_central_server(network, weights, writer, network.nb_initial_epochs)

    # Averaging models
    new_model = aggregate_models([client.trained_model for client in network.clients],
                     weights, network.clients[0].device)

    for i in range(network.nb_clients):
        load_new_model(network.clients[i].trained_model, new_model)
        assert equal(network.clients[i].trained_model, network.clients[0].trained_model), \
            (f"Models {network.clients[i].ID} are not equal.")

    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"=============== \tEpoch {synchronization_idx} ===============")

        # One pass of local training
        for  i in range(network.nb_clients):
            network.clients[i].continue_training(nb_of_local_epoch, synchronization_idx, single_batch=True)

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
        loss_accuracy_central_server(network, weights, writer, client.last_epoch)


def all_for_all_algo(network: Network, nb_of_synchronization: int = 5, inner_iterations: int = 1):
    print(f"--- nb_of_communication: {nb_of_synchronization} - inner_epochs {inner_iterations} ---")
    writer = SummaryWriter(
        log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/test/{network.dataset_name}_{network.algo_name}_'
                f'central_server')

    total_nb_points = np.sum([len(client.train_loader.dataset) for client in network.clients])
    fed_weights = [len(client.train_loader.dataset) / total_nb_points for client in network.clients]


    loss_accuracy_central_server(network, fed_weights, writer, network.nb_initial_epochs)

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


    for synchronization_idx in range(1, nb_of_synchronization + 1):
        print(f"===============\tEpoch {synchronization_idx}\t===============")

        # Compute weights len(client.train_loader.dataset) / total_nb_points
        # weights = [[int(i == j or rejection_test.aggreage_heter()[i][j] > 0.05) for j in range(network.nb_clients)]
        #                      for i in range(network.nb_clients)]
        weights = [[len(network.clients[j].train_loader.dataset) / total_nb_points for j in range(network.nb_clients)]
                                        for i in range(network.nb_clients)]
        # weights = normalize(weights, axis=1, norm='l1')
        # weights = weights @ weights.T
        print(f"At epoch {synchronization_idx}, weights are: \n {weights}.")

        for k in range(inner_iterations):
            gradients = []

            # Computing gradients on each clients.
            for client in network.clients:
                set_seed(client.last_epoch)
                # Single batch update before aggregation.
                gradient = gradient_step(client.train_loader, client.device, client.trained_model,
                                      client.criterion, client.optimizer, client.scheduler,
                                      client.last_epoch % len(client.train_loader))
                gradients.append(gradient)

            # Averaging gradient and updating models.
            for i in range(network.nb_clients):
                client = network.clients[i]
                # We plot the computed gradient on each client before their aggregation.
                for name, param in client.trained_model.named_parameters():
                    if param.grad is not None:
                        client.writer.add_histogram(f'{name}.grad', param.grad, client.last_epoch)

                aggregated_gradients = aggregate_gradients(gradients, weights[i])
                update_model(client.trained_model, aggregated_gradients, client.optimizer)

            for client in network.clients:
                client.last_epoch += 1
                write_train_val_test_performance(client.trained_model, client.device, client.train_loader,
                                                 client.val_loader, client.test_loader, client.criterion, client.metric,
                                                 client.ID, client.writer, client.last_epoch)

        rejection_test.reinitialize()
        compute_matrix_of_distances(rejection_pvalue, network, rejection_test, symetric_distance=True)

        loss_accuracy_central_server(network, fed_weights, writer, client.last_epoch)

        if synchronization_idx % 5 == 0:
            ### We compute the distance between clients.
            acceptance_test.reinitialize()
            compute_matrix_of_distances(acceptance_pvalue, network, acceptance_test, symetric_distance=False)

            rejection_test.reinitialize()
            compute_matrix_of_distances(rejection_pvalue, network, rejection_test, symetric_distance=True)

            plot_pvalues(acceptance_test, f"{synchronization_idx}")
            plot_pvalues(rejection_test, f"{synchronization_idx}")


        # # Compute weights
        # weights = [[1 / network.nb_clients for j in range(network.nb_clients)]
        #            for i in range(network.nb_clients)] # len(network.clients[j].train_loader.dataset)/ total_nb_points
        # # weights = [[ 1 / network.nb_clients for j in range(network.nb_clients)]
        # #            for i in range(network.nb_clients)]
        # # weights = [[int( i == j or acceptance_test.aggreage_heter()[i][j] > 0.05) for j in range(network.nb_clients)]
        # #                       for i in range(network.nb_clients)]
        # # weights = normalize(weights, axis=1, norm='l1')
        # # weights = weights @ weights.T
        # print(f"At epoch {epoch}, weights are: \n {weights}.")
        #
        # inner_epochs = 1
        # for k in range(inner_epochs):
        #     gradients = []
        #     for client in network.clients:
        #         print(f"last_epoch: {client.last_epoch}")
        #         set_seed(client.last_epoch)
        #         client.last_epoch += 1
        #         gradient = batch_training(client.train_loader, client.device, client.trained_model,
        #                                   client.criterion, client.optimizer, client.scheduler,
        #                                   ((epoch-1) * inner_epochs + k) % len(client.train_loader))
        #
        #         # gradient_step(
        #         # We divide by the total number of points s.t. each point have the same impact on the training
        #         gradients.append([g for g in gradient])
        #
        #     # Averaging gradient and updating models.
        #     for i in range(network.nb_clients):
        #         client = network.clients[i]
        #         aggregated_gradients = aggregate_gradients(gradients, weights[i])
        #         update_model(client.trained_model, aggregated_gradients, client.optimizer)
        #
        #         # assert equal(client.trained_model, network.clients[0].trained_model), f"Models {client.ID} are not equal."
        #
        #     for client in network.clients:
        #         client.last_epoch += 1
        #         # Writing logs.
        #         for name, param in client.trained_model.named_parameters():
        #             client.writer.add_histogram(f'{name}.weight', param, client.last_epoch)
        #             client.writer.add_histogram(f'{name}.grad', param.grad, client.last_epoch)
        #         log_performance("train", client.trained_model, client.device, client.train_loader, client.criterion,
        #                         client.metric, client.ID, client.writer,
        #                         client.last_epoch)
        #         log_performance("val", client.trained_model, client.device, client.val_loader, client.criterion,
        #                         client.metric, client.ID, client.writer,
        #                         client.last_epoch)
        #         log_performance("test", client.trained_model, client.device, client.test_loader, client.criterion,
        #                         client.metric, client.ID, client.writer,
        #                         client.last_epoch)
        #         loss_accuracy_central_server(network, fed_weights, writer, client.last_epoch)
        #         client.writer.close()
        #
        #

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
            weights = [int(i == j) for j in range(network.nb_clients)]
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

