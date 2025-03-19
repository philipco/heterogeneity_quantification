import glob
import os

from src.data.NetworkLoader import get_network
from src.utils.LoggingWriter import LoggingWriter
from src.utils.PlotUtilities import plot_values
from src.utils.Utilities import get_project_root, create_folder_if_not_existing

NB_RUN = 1

if __name__ == '__main__':

    # Recalculer l'erreur globale basée sur la moyenne et sur l'écart-type !

    dataset_name = "synth"

    assert dataset_name in ["exam_llm", "mnist", "cifar10", "heart_disease", "tcga_brca", "ixi", "liquid_asset", "synth"], \
        "Dataset not recognized."
    print(f"### ================== DATASET: {dataset_name} ================== ###")

    nb_initial_epochs = 0

    all_algos = ["all_for_one_ratio", "all_for_one", "all_for_one_loss", "all_for_one_pdtscl", "local", "all_for_all", "fednova", "fed"] #["gossip", "quantile", "fed", "all_for_all"]

    train_epochs, train_losses, train_accuracies = {algo: [] for algo in all_algos}, {algo: [] for algo in all_algos}, {algo: [] for algo in all_algos}
    test_epochs, test_losses, test_accuracies = {algo: [] for algo in all_algos}, {algo: [] for algo in all_algos}, {algo: [] for algo in all_algos}

    for algo_name in all_algos:
        assert algo_name in ["all_for_one_ratio", "all_for_one_pdtscl", "quantile", "gossip", "fed", "all_for_all", "all_for_one", "all_for_one_loss", "fednova", "local"], "Algorithm not recognized."
        print(f"--- ================== ALGO: {algo_name} ================== ---")

        root = get_project_root()
        pickle_folder = '{0}/pickle/{1}/{2}'.format(root, dataset_name, algo_name)

        # Use glob to find all files matching the pattern
        file_pattern = os.path.join(pickle_folder, 'logging_writer*.pkl')
        matching_files = glob.glob(file_pattern)

        # Extract the file names from the full paths
        file_names = [os.path.basename(file) for file in matching_files]
        for name in file_names:
            writer = LoggingWriter.load(pickle_folder, name)

            train_epochs[algo_name].append(writer.retrieve_information("train_accuracy")[0])
            train_accuracies[algo_name].append(writer.retrieve_information("train_accuracy")[1])
            train_losses[algo_name].append(writer.retrieve_information("train_loss")[1])

            test_epochs[algo_name].append(writer.retrieve_information("test_accuracy")[0])
            test_accuracies[algo_name].append(writer.retrieve_information("test_accuracy")[1])
            test_losses[algo_name].append(writer.retrieve_information("test_loss")[1])

    plot_values(train_epochs, train_accuracies, all_algos, 'Train_accuracy', dataset_name)
    plot_values(train_epochs, train_losses, all_algos, 'Train_loss', dataset_name, log=True)
    plot_values(test_epochs, test_accuracies, all_algos, 'Test_accuracy', dataset_name)
    plot_values(test_epochs, test_losses, all_algos, 'Test_loss', dataset_name, log=True)