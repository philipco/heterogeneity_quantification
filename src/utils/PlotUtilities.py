import matplotlib.pyplot as plt
import numpy as np

from src.data.DatasetConstants import BATCH_SIZE, STEP_SIZE
from src.utils.Utilities import get_project_root, create_folder_if_not_existing

COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:brown', 'tab:purple', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
MARKERS = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '*']

def plot_values(epochs, values, legends, metric_name, dataset_name, log=False):
    """
    Plots the given values against epochs using 'tab:colors'.

    Args:
        epochs (list or array): List of epoch numbers.
        values (list or array): List of correspplt.plot(epochs[algo_name][0], np.mean(values[algo_name], axis=0), marker='o', linestyle='-', color=COLORS[i], label=algo_name)
        onding values.
        name (str): Label for the y-axis.
    """
    plt.figure(figsize=(8, 5))
    i = 0
    for algo_name in legends:
        if log:
            avg_values = np.mean([np.log10(v) for v in values[algo_name]], axis=0)
            avg_values_var = np.std([np.log10(v) for v in values[algo_name]], axis=0)
            plt.plot(epochs[algo_name][0], avg_values, marker='o', linestyle='-', color=COLORS[i], label=algo_name)
            plt.fill_between(epochs[algo_name][0], avg_values - avg_values_var, avg_values + avg_values_var, alpha=0.2,
                             color=COLORS[i])

        else:
            avg_values = np.mean(values[algo_name], axis=0)
            avg_values_var = np.std(values[algo_name], axis=0)
            plt.plot(epochs[algo_name][0], avg_values, marker='o', linestyle='-', color=COLORS[i], label=algo_name)
            plt.fill_between(epochs[algo_name][0], avg_values - avg_values_var, avg_values + avg_values_var, alpha=0.2,
                             color=COLORS[i])

        i+=1
    plt.ylabel(metric_name)
    plt.legend()

    root = get_project_root()
    folder = f'{root}/pictures/convergence/{dataset_name}'
    create_folder_if_not_existing(folder)
    plt.savefig(f"{folder}/{metric_name}_b{BATCH_SIZE[dataset_name]}.pdf", bbox_inches='tight', dpi=600)

def plot_weights(weights, dataset_name, algo_name, name="weights"):
    nb_clients = len(weights)

    fig, axes = plt.subplots(nb_clients, 1, figsize=(6, nb_clients))
    for client_idx in range(nb_clients):
        ax = axes[client_idx]
        weight = weights[client_idx]

        # Plot each (row, column) entry through time
        iterations = range(len(weight))
        for c_idx in range(nb_clients):
            weight_to_plot = [weight[t][c_idx] for t in iterations]
            # if name.__eq__("weights"):
            #     ax.set_ylim([0, 1.5])
            ax.plot(iterations, weight_to_plot, label=f"Client i ← {c_idx}", color=COLORS[c_idx],
                    alpha=0.7, linestyle='-', marker=MARKERS[c_idx])

        ax.grid(True, linestyle='--', alpha=0.6)
        if client_idx == 0:
            ax.legend(ncol=2, fontsize=8, loc="best")

    plt.subplots_adjust(wspace=0, hspace=0)  # To remove the space between subplots

    root = get_project_root()
    folder = f'{root}/pictures/convergence/{dataset_name}'
    create_folder_if_not_existing(folder)
    plt.savefig(f"{folder}/{algo_name}_{name}_b{BATCH_SIZE[dataset_name]}.pdf", bbox_inches='tight', dpi=600)
