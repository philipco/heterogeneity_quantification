"""Created by Constantin Philippenko, 29th September 2022."""

import argparse

from src.data.Dataset import do_prediction_liquid_asset, load_liquid_dataset_test
from src.data.DatasetConstants import NB_EPOCHS
from src.data.NetworkLoader import get_network
from src.optim.Algo import federated_training, all_for_all_algo, \
    fednova_training, all_for_one_algo
from src.utils.PlotUtilities import plot_values, plot_weights
from src.utils.Utilities import get_path_to_datasets

NB_RUN = 1

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

    all_algos = ["All-for-one-bin", "All-for-one-cont", "Local", "FedAvg"]

    train_epochs, train_losses, train_accuracies = {algo: [] for algo in all_algos}, {algo: [] for algo in all_algos}, {
        algo: [] for algo in all_algos}
    test_epochs, test_losses, test_accuracies = {algo: [] for algo in all_algos}, {algo: [] for algo in all_algos}, {
        algo: [] for algo in all_algos}
    weights, ratio = {algo: [] for algo in all_algos}, {algo: [] for algo in all_algos}
    
    for algo_name in all_algos:
        assert algo_name in ["All-for-one-bin", "All-for-one-cont", "All-for-all", "Local", "FedAvg", "FedNova"], "Algorithm not recognized."
        print(f"--- ================== ALGO: {algo_name} ================== ---")

        network = get_network(dataset_name, algo_name)

        if algo_name == "FedAvg":
            federated_training(network, nb_of_synchronization=nb_epochs)
        if algo_name == "All-for-all":
            all_for_all_algo(network, nb_of_synchronization=nb_epochs, collab_based_on = "ratio")
        if algo_name == "Local":
            all_for_all_algo(network, nb_of_synchronization=nb_epochs, collab_based_on="local")
        if algo_name == "Fednova":
            fednova_training(network, nb_of_synchronization=nb_epochs)
        if algo_name == "All-for-one-bin":
            all_for_one_algo(network, nb_of_synchronization=nb_epochs, continuous=False)
        if algo_name == "All-for-one-cont":
            all_for_one_algo(network, nb_of_synchronization=nb_epochs, continuous=True)

        for client in network.clients:
            writer = client.writer

            train_epochs[algo_name].append(writer.retrieve_information("train_accuracy")[0])
            train_accuracies[algo_name].append(writer.retrieve_information("train_accuracy")[1])
            train_losses[algo_name].append(writer.retrieve_information("train_loss")[1])

            test_epochs[algo_name].append(writer.retrieve_information("test_accuracy")[0])
            test_accuracies[algo_name].append(writer.retrieve_information("test_accuracy")[1])
            test_losses[algo_name].append(writer.retrieve_information("test_loss")[1])

            weights[algo_name].append(writer.retrieve_histogram_information("weights")[1])
            ratio[algo_name].append(writer.retrieve_histogram_information("ratio")[1])


    plot_values(train_epochs, train_accuracies, all_algos, 'Train_accuracy', dataset_name)
    plot_values(train_epochs, train_losses, all_algos, 'Train_loss', dataset_name, log=True)
    plot_values(test_epochs, test_accuracies, all_algos, 'Test_accuracy', dataset_name)
    plot_values(test_epochs, test_losses, all_algos, 'Test_loss', dataset_name, log=True)

    for algo_name in all_algos:
        if algo_name in ["All-for-one-bin", "All-for-one-cont", "All-for-all"]:
            plot_weights(weights[algo_name], dataset_name, algo_name)#, x_axis=test_accuracies[algo_name])

    if dataset_name in ["liquid_asset"]:
        X_raw_train, X_raw_test, numerical_transformer = load_liquid_dataset_test(get_path_to_datasets())
        do_prediction_liquid_asset(network, X_raw_train, X_raw_test, numerical_transformer)
