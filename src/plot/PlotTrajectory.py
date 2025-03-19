"""Created by Constantin Philippenko, 18th February 2025."""

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data.DatasetConstants import BATCH_SIZE
from src.data.NetworkLoader import get_network
from src.optim.Algo import all_for_one_algo, all_for_all_algo, federated_training, fednova_training

from src.optim.Train import compute_loss_and_accuracy
from src.utils.PlotUtilities import plot_values
from src.utils.Utilities import get_project_root, create_folder_if_not_existing


def plot_level_set_with_gradients_pytorch(net, device, criterion, metric, data_loader, color, x_range=(-10, 10), y_range=(-10, 10),
                                          num_points=200):
    """
    Plots the level set of the loss function, points, and arrows representing the gradient at the corresponding points.

    Parameters:
    - model: nn.Module, the PyTorch neural network model.
    - criterion: loss function, the loss function to compute the gradients.
    - data_loader: DataLoader, the data loader containing the points.
    - x_range: tuple, the range of x values for the plot.
    - y_range: tuple, the range of y values for the plot.
    - num_points: int, the number of points to use in the meshgrid.
    """
    # Generate data for the level set
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Calculate the loss and gradients for each point in the data loader
    points = []

    with torch.no_grad():
        for i in range(num_points):
            for j in range(num_points):
                net.linear.weight.data = torch.tensor([[X[i, j]]]).to(torch.float32).to(device)
                net.linear.bias.data = torch.tensor([Y[i, j]]).to(torch.float32).to(device)
                Z[i, j] = compute_loss_and_accuracy(net, device, data_loader, criterion, metric, full_batch = True)[0]

    # points = np.concatenate(points, axis=0)
    # gradients = np.concatenate(gradients, axis=0)

    # Plot the level set
    plt.contour(X, Y, Z, levels=[0.05, 0.1, 0.5, 1, 2, 4, 8, 16], colors=color, alpha=0.75)

    # Plot the points
    # plt.scatter(points[:, 0], points[:, 1], color='red', label='Points')

    # Plot the arrows representing the gradients
    # for point, grad in zip(points, gradients):
    #     plt.arrow(point[0], point[1], grad[0], grad[1], head_width=0.5, head_length=0.5, fc='blue', ec='blue')


dataset_name = "synth"

NB_RUN = 1

if __name__ == '__main__':

    nb_initial_epochs = 0
    all_algos = ["all_for_one_ratio", "all_for_one", "all_for_one_loss", "all_for_one_pdtscl", "local", "all_for_all", "fednova", "fed"]

    train_epochs, train_losses, train_accuracies = {algo: [] for algo in all_algos}, {algo: [] for algo in all_algos}, {
        algo: [] for algo in all_algos}
    test_epochs, test_losses, test_accuracies = {algo: [] for algo in all_algos}, {algo: [] for algo in all_algos}, {
        algo: [] for algo in all_algos}

    for algo_name in all_algos:
        print(f"--- ================== ALGO: {algo_name} ================== ---")

        network = get_network(dataset_name, algo_name, nb_initial_epochs)

        for i in range(network.nb_clients):
            c = network.clients[i]
            plot_level_set_with_gradients_pytorch(c.trained_model, c.device, c.criterion, c.metric,
                                                  c.train_loader, color="tab:red" if i == 0 else "tab:blue")
            c.trained_model.linear.weight.data = torch.tensor([[-7.5]]).to(torch.float32).to(c.device)
            c.trained_model.linear.bias.data = torch.tensor([-7.5]).to(torch.float32).to(c.device)

        if algo_name == "fed":
            track_models = federated_training(network, nb_of_synchronization=20, keep_track=True)
        if algo_name == "fednova":
            track_models = fednova_training(network, nb_of_synchronization=20, keep_track=True)
        if algo_name == "all_for_all":
            track_models, track_gradients = all_for_all_algo(network, nb_of_synchronization=20, keep_track=True)
        if algo_name == "local":
            track_models, track_gradients = all_for_all_algo(network, nb_of_synchronization=20, collab_based_on = "local",
                                                             keep_track=True)
        if algo_name == "all_for_one":
            track_models, track_gradients = all_for_one_algo(network, nb_of_synchronization=20, collab_based_on = "grad",
                                                             keep_track=True)
        if algo_name == "all_for_one_loss":
            track_models, track_gradients = all_for_one_algo(network, nb_of_synchronization=20, collab_based_on = "loss",
                                                             keep_track=True)
        if algo_name == "all_for_one_pdtscl":
            track_models, track_gradients = all_for_one_algo(network, nb_of_synchronization=20, collab_based_on = "pdtscl",
                                                             keep_track=True)
        if algo_name == "all_for_one_ratio":
            track_models, track_gradients = all_for_one_algo(network, nb_of_synchronization=20, collab_based_on = "ratio",
                                                             keep_track=True)

        if not algo_name in ["fed", "fednova"]:
            for i in range(len(track_models)):
                X = [t[0] for t in track_models[i][:-2]]
                Y = [t[1] for t in track_models[i][:-2]]
                plt.scatter(X, Y, color="tab:red" if i == 0 else "tab:blue", marker="*")

                for j in range(0, len(X), 1):
                    plt.quiver(X[j], Y[j], -track_gradients[i][j][0], -track_gradients[i][j][1],
                               angles='xy', scale_units='xy', scale=1, color="tab:red" if i == 0 else "tab:blue", alpha=0.5,
                               width=0.005)
        else:
            X = [t[0] for t in track_models]
            Y = [t[1] for t in track_models]
            plt.scatter(X, Y, color="tab:red", marker="*")

        for client in network.clients:
            writer = client.writer

            train_epochs[algo_name].append(writer.retrieve_information("train_accuracy")[0])
            train_accuracies[algo_name].append(writer.retrieve_information("train_accuracy")[1])
            train_losses[algo_name].append(writer.retrieve_information("train_loss")[1])

            test_epochs[algo_name].append(writer.retrieve_information("test_accuracy")[0])
            test_accuracies[algo_name].append(writer.retrieve_information("test_accuracy")[1])
            test_losses[algo_name].append(writer.retrieve_information("test_loss")[1])

        # Saving the writer as pkl files.
        root = get_project_root()
        pickle_folder = '{0}/pickle/{1}/{2}'.format(root, dataset_name, algo_name)
        create_folder_if_not_existing(pickle_folder)
        network.writer.save(f"{pickle_folder}", "logging_writer_central.pkl")
        for client in network.clients:
            client.writer.save(f"{pickle_folder}", f"logging_writer_{client.ID}.pkl")

        # Saving the level set figure.
        create_folder_if_not_existing('{0}/pictures/convergence'.format(root))
        folder = f'{root}/pictures/convergence/{dataset_name}_{algo_name}_b{BATCH_SIZE[dataset_name]}.pdf'
        plt.savefig(folder, dpi=600, bbox_inches='tight')
        plt.close()

    plot_values(train_epochs, train_accuracies, all_algos, 'Train_accuracy', dataset_name)
    plot_values(train_epochs, train_losses, all_algos, 'Train_loss', dataset_name, log=False)
    plot_values(test_epochs, test_accuracies, all_algos, 'Test_accuracy', dataset_name)
    plot_values(test_epochs, test_losses, all_algos, 'Test_loss', dataset_name, log=False)

