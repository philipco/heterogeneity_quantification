"""
Main script to run all experiments provided in the paper.
"""

import argparse

import torch

from src.data.LiquidAssetDataset import do_prediction_liquid_asset, load_liquid_dataset_test
from src.data.DatasetConstants import NB_EPOCHS
from src.data.Network import get_network
from src.optim.Algo import fedavg_training, all_for_all_algo, all_for_one_algo, fednova_training, cobo_algo, ditto_algo, \
    wga_bc_algo
from src.utils.PlotUtilities import plot_values, plot_weights
from src.utils.Utilities import get_path_to_datasets

if __name__ == '__main__':

    train_epochs, train_losses, train_accuracies = {}, {}, {}
    test_epochs, test_losses, test_accuracies = {}, {}, {}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset.",
        required=True,
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name

    assert dataset_name in ["exam_llm", "mnist", "mnist_iid", "cifar10", "cifar10_iid", "heart_disease", "tcga_brca", "ixi", "liquid_asset",
                            "synth", "synth_complex"], "Dataset not recognized."
    print(f"### ================== DATASET: {dataset_name} ================== ###")

    nb_epochs = NB_EPOCHS[dataset_name]

    # We double the torch precision because for synthetic data, the loss is sometimes set to 0 while it should be 10^-12.
    if "synth" in dataset_name:
        torch.set_default_dtype(torch.float64)

    all_algos = ["All-for-one-bin", "WGA-BC", "Local", "FedAvg"]#, "Ditto", "Cobo", "WGA-BC"]
    all_seeds = [127, 496, 1729] # Mersenne number, Perfect number, Ramanujan number

    def dict(all_algos, all_seeds):
        return {algo: {s: [] for s in all_seeds} for algo in all_algos}

    train_epochs, train_losses, train_accuracies = dict(all_algos, all_seeds), dict(all_algos, all_seeds), dict(all_algos, all_seeds)
    test_epochs, test_losses, test_accuracies = dict(all_algos, all_seeds), dict(all_algos, all_seeds), dict(all_algos, all_seeds)
    weights, ratio = dict(all_algos, all_seeds), dict(all_algos, all_seeds)
    
    for algo_name in all_algos:

        assert algo_name in ["All-for-one-bin", "All-for-one-cont", "All-for-all", "Local", "FedAvg", "FedNova",
                             "Cobo", "Ditto", "WGA-BC"], "Algorithm not recognized."
        print(f"--- ================== ALGO: {algo_name} ================== ---")

        for seed in all_seeds:
            network = get_network(dataset_name, algo_name, seed)

            if algo_name == "FedAvg":
                fedavg_training(network, nb_of_synchronization=nb_epochs)
            elif algo_name == "All-for-all":
                all_for_all_algo(network, nb_of_synchronization=nb_epochs, collab_based_on="ratio")
            elif algo_name == "Local":
                all_for_all_algo(network, nb_of_synchronization=nb_epochs, collab_based_on="local")
            elif algo_name == "Fednova":
                fednova_training(network, nb_of_synchronization=nb_epochs)
            elif algo_name == "All-for-one-bin":
                all_for_one_algo(network, nb_of_synchronization=nb_epochs, continuous=False)
            elif algo_name == "All-for-one-cont":
                all_for_one_algo(network, nb_of_synchronization=nb_epochs, continuous=True)
            elif algo_name == "Cobo":
                cobo_algo(network, nb_of_synchronization=nb_epochs)
            elif algo_name == "Ditto":
                ditto_algo(network, nb_of_synchronization=nb_epochs)
            elif algo_name == "WGA-BC":
                wga_bc_algo(network, nb_of_synchronization=nb_epochs)

            for client in network.clients:
                writer = client.writer

                train_epochs[algo_name][seed].append(writer.retrieve_information("train_accuracy")[0])
                train_accuracies[algo_name][seed].append(writer.retrieve_information("train_accuracy")[1])
                train_losses[algo_name][seed].append(writer.retrieve_information("train_loss")[1])

                test_epochs[algo_name][seed].append(writer.retrieve_information("test_accuracy")[0])
                test_accuracies[algo_name][seed].append(writer.retrieve_information("test_accuracy")[1])
                test_losses[algo_name][seed].append(writer.retrieve_information("test_loss")[1])

                weights[algo_name][seed].append(writer.retrieve_histogram_information("weights")[1])
                ratio[algo_name][seed].append(writer.retrieve_histogram_information("ratio")[1])

    plot_values(train_epochs, train_accuracies, all_algos, 'Train accuracy', dataset_name)
    plot_values(train_epochs, train_losses, all_algos, 'log(Train loss)', dataset_name, log=True)
    plot_values(test_epochs, test_accuracies, all_algos, 'Test accuracy', dataset_name)
    plot_values(test_epochs, test_losses, all_algos, 'log(Test loss)', dataset_name, log=True)

    for algo_name in all_algos:
        if algo_name in ["All-for-one-bin", "All-for-one-cont", "All-for-all", "Cobo"]:
            plot_weights(weights[algo_name][all_seeds[0]], dataset_name, algo_name)#, x_axis=test_accuracies[algo_name])

    if dataset_name in ["liquid_asset"]:
        X_raw_train, X_raw_test, numerical_transformer = load_liquid_dataset_test(get_path_to_datasets())
        do_prediction_liquid_asset(network, X_raw_train, X_raw_test, numerical_transformer)
