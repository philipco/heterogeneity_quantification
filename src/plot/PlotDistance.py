"""Created by Constantin Philippenko, 30th September 2022."""
from typing import List

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.quantif.Metrics import Metrics
from src.utils.Utilities import create_folder_if_not_existing, get_project_root
from src.utils.UtilitiesNumpy import remove_diagonal, create_matrix_with_zeros_diagonal_from_array

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def scaling(caliber, matrix_to_plot: List[np.array], scaler):
    # transform = scaler.fit_transform(caliber)
    # m0, m1 = transform[:len(transform) // 2], transform[len(transform) // 2:]
    scaler.fit(caliber)
    m0 = scaler.transform(remove_diagonal(matrix_to_plot[0], False).reshape(-1, 1))
    m1 = scaler.transform(remove_diagonal(matrix_to_plot[1], False).reshape(-1, 1))

    matrix_to_plot[0] = create_matrix_with_zeros_diagonal_from_array(m0)
    matrix_to_plot[1] = create_matrix_with_zeros_diagonal_from_array(m1)

    return matrix_to_plot


def plot_distance(metrics: Metrics) -> None:
    fig, axes = plt.subplots(1, 1)

    matrix_to_plot = np.mean(metrics.distances_heter, axis=0)

    if not (np.diag(matrix_to_plot[0]) < 10e-3).all():
        print("WARNING: The diagonal of the iid distance's matrix is not null")
    if not (np.diag(matrix_to_plot[1]) < 10e-3).all():
        print("WARNING: The diagonal of the non-iid distance's matrix is not null")

    if False:
        print("Standard scaling before plotting ...")
        benchmark_for_scaling = np.array(remove_diagonal(matrix_to_plot[0], symmetric_matrix=False)).reshape(-1, 1)
        matrix_to_plot = scaling(benchmark_for_scaling, matrix_to_plot, StandardScaler())

    clients_order = np.arange(metrics.nb_of_clients)

    one_to_one_min = min(matrix_to_plot.flatten())
    one_to_one_max = max(matrix_to_plot.flatten())

    cmap = plt.cm.Reds
    cmap.set_bad((204 / 255, 204 / 255, 204 / 255, 1))
    kwargs = dict(origin='lower', vmin=one_to_one_min, vmax=one_to_one_max, aspect="equal", cmap=cmap)

    # We set the diagonal at nan.
    # matrix_to_plot[0][np.arange(metrics.nb_of_clients), np.arange(metrics.nb_of_clients)] = np.nan
    # matrix_to_plot[1][np.arange(metrics.nb_of_clients), np.arange(metrics.nb_of_clients)] = np.nan

    x_non_iid = 0
    total_nb_of_points = np.sum(metrics.nb_of_points_by_clients)
    xticks, yticks = [], []
    for x in range(metrics.nb_of_clients):
        y_non_iid = 0
        for y in range(metrics.nb_of_clients):
            # The size of each row/column is depedent of the number of point on the client.
            # data1 = matrix_to_plot[1][x:x + 1, y:y + 1]
            data0 = matrix_to_plot[x:x + 1, y:y + 1]
            size_x = metrics.nb_of_points_by_clients[clients_order[x]] / total_nb_of_points * metrics.nb_of_clients
            size_y = metrics.nb_of_points_by_clients[clients_order[y]] / total_nb_of_points * metrics.nb_of_clients
            im = axes.imshow(data0, extent=[x_non_iid, x_non_iid + size_x, y_non_iid, y_non_iid + size_y],
                           **kwargs)

            y_non_iid += size_y
            if len(yticks) != metrics.nb_of_clients:
                yticks.append(y_non_iid + size_y / 2)
        xticks.append(x_non_iid + size_x / 2)
        x_non_iid += size_x

    axes.set_ylim(0, metrics.nb_of_clients)
    axes.set_xlim(0, metrics.nb_of_clients)
    axes.set_yticks(xticks)
    axes.set_xticks(xticks)
    axes.set_xticklabels(clients_order, fontsize=15)
    axes.set_yticklabels(clients_order, fontsize=15)

    cbar = fig.colorbar(im, ax=axes, shrink=0.5)
    cbar.formatter.set_powerlimits((0, 0))

    root = get_project_root()
    create_folder_if_not_existing('{0}/pictures/{1}'.format(root, metrics.dataset_name))
    plt.savefig('{0}/pictures/{1}/{2}.pdf'.format(root, metrics.dataset_name, metrics.distance_name),
                bbox_inches='tight', dpi=600)
    np.savetxt('{0}/pictures/{1}/{2}-heter.txt'.format(root, metrics.dataset_name, metrics.distance_name),
               matrix_to_plot, delimiter=',')

def plot_distance_scaled_with_iid(metrics: Metrics) -> None:

    fig, axes = plt.subplots(1, 2)

    matrix_to_plot = [np.mean(metrics.distances_iid, axis=0), np.mean(metrics.distance_heter, axis=0)]

    if not (np.diag(matrix_to_plot[0]) < 10e-3).all():
        print("WARNING: The diagonal of the iid distance's matrix is not null")
    if not (np.diag(matrix_to_plot[1]) < 10e-3).all():
        print("WARNING: The diagonal of the non-iid distance's matrix is not null")

    if False:
        print("Standard scaling before plotting ...")
        benchmark_for_scaling = np.array(remove_diagonal(matrix_to_plot[0], symmetric_matrix=False)).reshape(-1, 1)
        matrix_to_plot = scaling(benchmark_for_scaling, matrix_to_plot, StandardScaler())

    clients_order = np.arange(metrics.nb_of_clients)

    one_to_one_min = min(min(matrix_to_plot[0].flatten()),
                         min(matrix_to_plot[1].flatten()))
    one_to_one_max = max(max(matrix_to_plot[0].flatten()),
                         max(matrix_to_plot[1].flatten()))

    cmap = plt.cm.Reds
    cmap.set_bad((204 / 255, 204 / 255, 204 / 255, 1))
    kwargs = dict(origin='lower', vmin=one_to_one_min, vmax=one_to_one_max, aspect="equal", cmap=cmap)

    # We set the diagonal at nan.
    # matrix_to_plot[0][np.arange(metrics.nb_of_clients), np.arange(metrics.nb_of_clients)] = np.nan
    # matrix_to_plot[1][np.arange(metrics.nb_of_clients), np.arange(metrics.nb_of_clients)] = np.nan

    x_non_iid = 0
    total_nb_of_points = np.sum(metrics.nb_of_points_by_clients)
    xticks, yticks = [], []
    for x in range(metrics.nb_of_clients):
        y_non_iid = 0
        for y in range(metrics.nb_of_clients):
            # The size of each row/column is depedent of the number of point on the client.
            data1 = matrix_to_plot[1][x:x + 1, y:y + 1]
            data0 = matrix_to_plot[0][x:x + 1, y:y + 1]
            size_x = metrics.nb_of_points_by_clients[clients_order[x]] / total_nb_of_points * metrics.nb_of_clients
            size_y = metrics.nb_of_points_by_clients[clients_order[y]] / total_nb_of_points * metrics.nb_of_clients
            axes[0].imshow(data0, extent=[x_non_iid, x_non_iid + size_x, y_non_iid, y_non_iid + size_y],
                           **kwargs)
            im1 = axes[1].imshow(data1, extent=[x_non_iid, x_non_iid + size_x, y_non_iid, y_non_iid + size_y],
                                 **kwargs)

            y_non_iid += size_y
            if len(yticks) != metrics.nb_of_clients:
                yticks.append(y_non_iid + size_y / 2)
        xticks.append(x_non_iid + size_x / 2)
        x_non_iid += size_x

    for ax in axes[:2]:
        ax.set_ylim(0, metrics.nb_of_clients)
        ax.set_xlim(0, metrics.nb_of_clients)
        ax.set_yticks(xticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels(clients_order, fontsize=15)
        ax.set_yticklabels(clients_order, fontsize=15)

    cbar = fig.colorbar(im1, ax=axes[:], shrink=0.5)
    cbar.formatter.set_powerlimits((0, 0))

    root = get_project_root()
    create_folder_if_not_existing('{0}/pictures/{1}'.format(root, metrics.dataset_name))
    plt.savefig('{0}/pictures/{1}/{2}.pdf'.format(root, metrics.dataset_name, metrics.distance_name),
                bbox_inches='tight', dpi=600)
    np.savetxt('{0}/pictures/{1}/{2}-iid.txt'.format(root, metrics.dataset_name, metrics.distance_name),
               matrix_to_plot[0], delimiter=',')
    np.savetxt('{0}/pictures/{1}/{2}-heter.txt'.format(root, metrics.dataset_name, metrics.distance_name),
               matrix_to_plot[1], delimiter=',')