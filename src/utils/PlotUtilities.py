import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

from src.data.DatasetConstants import BATCH_SIZE, STEP_SIZE, MOMENTUM
from src.utils.Utilities import get_project_root, create_folder_if_not_existing

COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:brown', 'tab:green', 'tab:purple']
MARKERS = ['o', 's', 'D', '^', 'v', '<']

FONTSIZE = 25

def plot_values(epochs, values, legends, metric_name, dataset_name, log=False):
    """
    Plots the given values against epochs using 'tab:colors'.

    Args:
        epochs (list or array): List of epoch numbers.
        values (list or array): List of correspplt.plot(epochs[algo_name][0], np.mean(values[algo_name], axis=0), marker='o', linestyle='-', color=COLORS[i], label=algo_name)
        onding values.
        name (str): Label for the y-axis.
    """
    plt.figure(figsize=(9, 6))
    i = 0
    for algo_name in legends:
        if log:
            avg_values = np.mean([np.log10(v) for v in values[algo_name]], axis=0)
            avg_values_var = np.std([np.log10(v) for v in values[algo_name]], axis=0)
            plt.plot(epochs["Local"][0], avg_values, linestyle='-', color=COLORS[i], label=algo_name, linewidth=5)
            plt.fill_between(epochs["Local"][0], avg_values - avg_values_var, avg_values + avg_values_var, alpha=0.2,
                             color=COLORS[i])

        else:
            avg_values = np.mean(values[algo_name], axis=0)
            avg_values_var = np.std(values[algo_name], axis=0)
            plt.plot(epochs["Local"][0], avg_values, linestyle='-', color=COLORS[i], label=algo_name, linewidth=5)
            plt.fill_between(epochs["Local"][0], avg_values - avg_values_var, avg_values + avg_values_var, alpha=0.2,
                             color=COLORS[i])

        i+=1

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.ylabel(metric_name, fontsize=FONTSIZE)
    plt.xlabel("Number of epochs", fontsize=FONTSIZE)
    if (metric_name == "log(Test loss)" and dataset_name == "mnist"):
        loc = "upper right"
    elif (metric_name == "Test accuracy" and dataset_name == "mnist"):
        loc = "lower right"
    else:
        loc = "lower left"
    if dataset_name in ["mnist", "synth"]:
        plt.legend(fontsize=FONTSIZE, loc=loc)


    root = get_project_root()
    folder = f'{root}/pictures/{dataset_name}'
    create_folder_if_not_existing(folder)
    plt.savefig(f"{folder}/{metric_name}_b{BATCH_SIZE[dataset_name]}_LR{STEP_SIZE[dataset_name]}_m{MOMENTUM[dataset_name]}.pdf",
                bbox_inches='tight', dpi=600)

    # Print the LaTeX table
    print("\\begin{tabular}{|c|c|}")
    print("\\hline")
    print(f"Algorithm & {metric_name} \\\\")
    print("\\hline")
    for algo_name in legends:
        if log:
            print(f"{algo_name} & {np.mean([np.log10(v) for v in values[algo_name]], axis=0)[-1]:.4f} \\\\")
        else:
            print(f"{algo_name} & {np.mean([v for v in values[algo_name]], axis=0)[-1]:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")

def plot_weights(weights, dataset_name, algo_name, name="weights", x_axis=None):
    nb_clients = len(weights)

    fig, axes = plt.subplots(min(nb_clients, len(MARKERS)), 1, figsize=(6, min(nb_clients, len(MARKERS))))
    for client_idx in range(min(nb_clients, len(MARKERS))):
        ax = axes[client_idx]
        weight = weights[client_idx]

        # Plot each (row, column) entry through time
        iterations = range(len(weight))
        for c_idx in range(min(nb_clients, len(MARKERS))):
            weight_to_plot = [weight[t][c_idx] for t in iterations]
            if x_axis:
                sorted_indices = np.argsort(x_axis[c_idx][:-1])
                sorted_x_axis = np.array(x_axis[c_idx][:-1])[sorted_indices]
                sorted_weight_to_plot = np.array(weight_to_plot)[sorted_indices]
                ax.plot(np.log10(sorted_x_axis), sorted_weight_to_plot, label=f"Client i ← {c_idx}", color=COLORS[c_idx],
                        alpha=0.7, linestyle='-', marker=MARKERS[c_idx])
            else:
                ax.plot(iterations, weight_to_plot, label=f"Client i ← {c_idx}", color=COLORS[c_idx],
                        alpha=0.7, linestyle='-', marker=MARKERS[c_idx])

        ax.grid(True, linestyle='--', alpha=0.6)
        if client_idx == 0:
            ax.legend(ncol=2, fontsize=FONTSIZE, loc="best")

    plt.subplots_adjust(wspace=0, hspace=0)  # To remove the space between subplots

    root = get_project_root()
    folder = f'{root}/pictures/{dataset_name}'
    create_folder_if_not_existing(folder)
    plt.savefig(f"{folder}/{algo_name}_{name}_b{BATCH_SIZE[dataset_name]}_LR{STEP_SIZE[dataset_name]}_m{MOMENTUM[dataset_name]}.pdf",
                bbox_inches='tight', dpi=600)
