import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data.DatasetConstants import BATCH_SIZE
from src.data.Network import get_network
from src.optim.Algo import all_for_one_algo, all_for_all_algo, fedavg_training, fednova_training, cobo_algo, ditto_algo

from src.utils.PlotUtilities import plot_values, plot_weights
from src.utils.Utilities import get_project_root, create_folder_if_not_existing

COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

torch.set_default_dtype(torch.float64)

def plot_level_set_with_gradients_pytorch(data_loader, x_range=(-10, 10), y_range=(-10, 10), num_points=200):
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

    with torch.no_grad():
        for i in range(num_points):
            for j in range(num_points):
                model_shift = data_loader.dataset.true_theta  - torch.tensor([[X[i, j], Y[i, j]]]).to(torch.float32)
                Z[i, j] = model_shift @ data_loader.dataset.covariance @ model_shift.T #+ data_loader.dataset.noise_std**2
    return X, Y, Z


dataset_name = "synth"

NB_RUN = 1
NB_EPOCHS = 30

if __name__ == '__main__':

    nb_initial_epochs = 0
    all_algos = ["All-for-one-cont", "Ditto", "Cobo", "Local"]
    all_seeds = [127]


    def dict(all_algos, all_seeds):
        return {algo: {s: [] for s in all_seeds} for algo in all_algos}


    train_epochs, train_losses, train_accuracies = dict(all_algos, all_seeds), dict(all_algos, all_seeds), dict(
        all_algos, all_seeds)
    test_epochs, test_losses, test_accuracies = dict(all_algos, all_seeds), dict(all_algos, all_seeds), dict(all_algos,
                                                                                                             all_seeds)
    weights, ratio = dict(all_algos, all_seeds), dict(all_algos, all_seeds)

    client_X, client_Y, client_Z = None, None, None
    for algo_name in all_algos:
        print(f"--- ================== ALGO: {algo_name} ================== ---")

        network = get_network(dataset_name, algo_name, initial_seed=127)
        if client_X is None:
            client_X, client_Y, client_Z = ([None for _ in network.clients] for _ in range(3))

        for i in range(network.nb_clients):
            c = network.clients[i]
            # Plot the level set
            if client_X[i] is None:
                client_X[i], client_Y[i], client_Z[i] = plot_level_set_with_gradients_pytorch(c.train_loader)
            plt.contour(client_X[i], client_Y[i], client_Z[i], levels=[0.01, 0.1, 1, 3, 7, 9, 15, 30], alpha=0.75)
                        #colors=COLORS[i], alpha=0.75)

            c.trained_model.linear.weight.data = torch.tensor([[0, 0]]).to(torch.double).to(c.device)

        if algo_name == "FedAvg":
            fedavg_training(network, nb_of_synchronization=NB_EPOCHS)
        if algo_name == "All-for-all":
            track_models, track_gradients = all_for_all_algo(network, nb_of_synchronization=NB_EPOCHS,
                                                             collab_based_on="ratio", keep_track=True)
        if algo_name == "Local":
            track_models, track_gradients = all_for_all_algo(network, nb_of_synchronization=NB_EPOCHS,
                                                             collab_based_on="local", keep_track=True)
        if algo_name == "Fednova":
            fednova_training(network, nb_of_synchronization=NB_EPOCHS)
        if algo_name == "All-for-one-bin":
            track_models, track_gradients = all_for_one_algo(network, nb_of_synchronization=NB_EPOCHS, continuous=False,
                                                             keep_track=True)
        if algo_name == "All-for-one-cont":
            track_models, track_gradients = all_for_one_algo(network, nb_of_synchronization=NB_EPOCHS, continuous=True,
                                                             keep_track=True)
        elif algo_name == "Cobo":
            track_models, track_gradients = cobo_algo(network, nb_of_synchronization=NB_EPOCHS, keep_track=True)
        elif algo_name == "Ditto":
            track_models, track_gradients = ditto_algo(network, nb_of_synchronization=NB_EPOCHS, keep_track=True)

        if not algo_name in ["FedAvg", "Fednova"]:
            for i in range(len(track_models)):
                model_X = [t[0][0] for t in track_models[i][:-1]]
                model_Y = [t[0][1] for t in track_models[i][:-1]]
                plt.scatter(model_X, model_Y, marker="*")#, color=COLORS[i]
                print(len(model_X))
                for j in range(0, len(model_X), 1):
                    plt.quiver(model_X[j], model_Y[j], -track_gradients[i][j][0][0], -track_gradients[i][j][0][1],
                               angles='xy', scale_units='xy', scale=1, alpha=0.5, width=0.005)#, color=COLORS[i])
        else:
            model_X = [t[0][0] for t in track_models[i][:-2]]
            model_Y = [t[0][1] for t in track_models[i][:-2]]
            plt.scatter(model_X, model_Y, color="tab:purple", marker="*")

        for client in network.clients:
            writer = client.writer

            train_epochs[algo_name][all_seeds[0]].append(writer.retrieve_information("train_accuracy")[0])
            train_accuracies[algo_name][all_seeds[0]].append(writer.retrieve_information("train_accuracy")[1])
            train_losses[algo_name][all_seeds[0]].append(writer.retrieve_information("train_loss")[1])

            test_epochs[algo_name][all_seeds[0]].append(writer.retrieve_information("test_accuracy")[0])
            test_accuracies[algo_name][all_seeds[0]].append(writer.retrieve_information("test_accuracy")[1])
            test_losses[algo_name][all_seeds[0]].append(writer.retrieve_information("test_loss")[1])

            weights[algo_name][all_seeds[0]].append(writer.retrieve_histogram_information("weights")[1])
            ratio[algo_name][all_seeds[0]].append(writer.retrieve_histogram_information("ratio")[1])


        root = get_project_root()
        # Saving the level set figure.
        create_folder_if_not_existing('{0}/pictures/convergence'.format(root))
        folder = f'{root}/pictures/{dataset_name}_{algo_name}_b{BATCH_SIZE[dataset_name]}.pdf'
        plt.savefig(folder, dpi=600, bbox_inches='tight')
        plt.close()

    plot_values(train_epochs, train_accuracies, all_algos, 'Train_accuracy', dataset_name)
    plot_values(train_epochs, train_losses, all_algos, 'Train_loss', dataset_name, log=True)
    plot_values(test_epochs, test_accuracies, all_algos, 'Test_accuracy', dataset_name)
    plot_values(test_epochs, test_losses, all_algos, 'Test_loss', dataset_name, log=True)

    for algo_name in all_algos:
        if algo_name not in ["fed", "fednova"]:
            plot_weights(weights[algo_name][all_seeds[0]], dataset_name, algo_name)

