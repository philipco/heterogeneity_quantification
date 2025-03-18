"""Created by Constantin Philippenko, 29th September 2022."""

import argparse

from src.data.Dataset import do_prediction_liquid_asset, load_liquid_dataset_test
from src.data.NetworkLoader import get_network
from src.optim.Algo import fedquantile_training, federated_training, gossip_training, all_for_all_algo, \
    fednova_training, all_for_one_algo
from src.utils.PlotUtilities import plot_values
from src.utils.Utilities import get_path_to_datasets, get_project_root, create_folder_if_not_existing

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

    assert dataset_name in ["exam_llm", "mnist", "cifar10", "heart_disease", "tcga_brca", "ixi", "liquid_asset", "synth"], \
        "Dataset not recognized."
    print(f"### ================== DATASET: {dataset_name} ================== ###")

    nb_initial_epochs = 0

    all_algos = ["all_for_one_ratio"] #, "all_for_one", "all_for_one_loss", "all_for_one_pdtscl", "local", "all_for_all", "fednova", "fed"] #["gossip", "quantile", "fed", "all_for_all"]
    for algo_name in all_algos:
        assert algo_name in ["all_for_one_ratio", "all_for_one_pdtscl", "quantile", "gossip", "fed", "all_for_all", "all_for_one", "all_for_one_loss", "fednova", "local"], "Algorithm not recognized."
        print(f"--- ================== ALGO: {algo_name} ================== ---")

        network = get_network(dataset_name, algo_name, nb_initial_epochs)

        if algo_name == "quantile":
            fedquantile_training(network)
        if algo_name == "gossip":
            gossip_training(network)
        if algo_name == "fed":
            federated_training(network, nb_of_synchronization=20)
        if algo_name == "all_for_all":
            all_for_all_algo(network, nb_of_synchronization=20)
        if algo_name == "local":
            all_for_all_algo(network, nb_of_synchronization=20, collab_based_on="local")
        if algo_name == "fednova":
            fednova_training(network, nb_of_synchronization=20)
        if algo_name == "all_for_one":
            all_for_one_algo(network, nb_of_synchronization=20)
        if algo_name == "all_for_one_loss":
            all_for_one_algo(network, nb_of_synchronization=20, collab_based_on="loss")
        if algo_name == "all_for_one_pdtscl":
            all_for_one_algo(network, nb_of_synchronization=20, collab_based_on="pdtscl")
        if algo_name == "all_for_one_ratio":
            all_for_one_algo(network, nb_of_synchronization=20, collab_based_on="ratio")

        train_epochs[algo_name] = network.writer.retrieve_information("train_accuracy")[0]
        train_accuracies[algo_name] = network.writer.retrieve_information("train_accuracy")[1]
        train_losses[algo_name] = network.writer.retrieve_information("train_loss")[1]

        test_epochs[algo_name] = network.writer.retrieve_information("test_accuracy")[0]
        test_accuracies[algo_name] = network.writer.retrieve_information("test_accuracy")[1]
        test_losses[algo_name] = network.writer.retrieve_information("test_loss")[1]

        root = get_project_root()
        pickle_folder = '{0}/pickle/{1}/{2}'.format(root, dataset_name, algo_name)
        create_folder_if_not_existing(pickle_folder)
        network.writer.save(f"{pickle_folder}_central")
        for client in network.clients:
            client.writer.save(f"{pickle_folder}_{client.ID}")

    plot_values(train_epochs, train_accuracies, all_algos, 'Train_accuracy', dataset_name)
    plot_values(train_epochs, train_losses, all_algos, 'Train_loss', dataset_name)
    plot_values(test_epochs, test_accuracies, all_algos, 'Test_accuracy', dataset_name)
    plot_values(test_epochs, test_losses, all_algos, 'Test_loss', dataset_name)

    if dataset_name in ["liquid_asset"]:
        X_raw_train, X_raw_test, numerical_transformer = load_liquid_dataset_test(get_path_to_datasets())
        do_prediction_liquid_asset(network, X_raw_train, X_raw_test, numerical_transformer)
