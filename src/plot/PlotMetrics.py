import glob
import os

from src.utils.LoggingWriter import LoggingWriter
from src.utils.PlotUtilities import plot_values, plot_weights
from src.utils.Utilities import get_project_root


if __name__ == '__main__':

    dataset_name = "synth"

    assert dataset_name in ["exam_llm", "mnist", "mnist_iid", "cifar10", "cifar10_iid", "heart_disease", "tcga_brca", "ixi", "liquid_asset",
                            "synth", "synth_complex"], "Dataset not recognized."
    print(f"### ================== DATASET: {dataset_name} ================== ###")

    nb_initial_epochs = 0

    all_algos = ["All-for-one-bin", "All-for-one-cont", "Local", "FedAvg", "Cobo", "Ditto", "WGA-BC", "APFL"]
    all_seeds = [127, 496, 1729] # Mersenne number, Perfect number, Ramanujan number

    def dict(all_algos, all_seeds):
        return {algo: {s: [] for s in all_seeds} for algo in all_algos}

    train_epochs, train_losses, train_accuracies = dict(all_algos, all_seeds), dict(all_algos, all_seeds), dict(all_algos, all_seeds)
    test_epochs, test_losses, test_accuracies = dict(all_algos, all_seeds), dict(all_algos, all_seeds), dict(all_algos, all_seeds)
    weights, ratio = dict(all_algos, all_seeds), dict(all_algos, all_seeds)

    for algo_name in all_algos:
        assert algo_name in ["All-for-one-bin", "All-for-one-cont", "All-for-all", "Local", "FedAvg", "FedNova",
                             "Ditto", "Cobo", "WGA-BC", "APFL"], \
            "Algorithm not recognized."
        print(f"--- ================== ALGO: {algo_name} ================== ---")

        for seed in all_seeds:

            root = get_project_root()
            pickle_folder = '{0}/pickle/{1}/{2}/{3}'.format(root, dataset_name, algo_name, seed)

            # Use glob to find all files matching the pattern
            file_pattern = os.path.join(pickle_folder, 'logging_writer*.pkl')
            matching_files = glob.glob(file_pattern)

            # Extract the file names from the full paths
            file_names = sorted([os.path.basename(file) for file in matching_files])
            for name in file_names:
                if name == 'logging_writer_central.pkl':
                    continue

                writer = LoggingWriter.load(pickle_folder, name)

                train_epochs[algo_name][seed].append(writer.retrieve_information("train_accuracy")[0])
                train_accuracies[algo_name][seed].append(writer.retrieve_information("train_accuracy")[1])
                train_losses[algo_name][seed].append(writer.retrieve_information("train_loss")[1])

                test_epochs[algo_name][seed].append(writer.retrieve_information("test_accuracy")[0])
                test_accuracies[algo_name][seed].append(writer.retrieve_information("test_accuracy")[1])
                test_losses[algo_name][seed].append(writer.retrieve_information("test_loss")[1])

                weights[algo_name][seed].append(writer.retrieve_histogram_information("weights")[1])
                ratio[algo_name][seed].append(writer.retrieve_histogram_information("ratio")[1])

            if algo_name not in ["FedAvg", "FedNova", "WGA-BC"]:
                plot_weights(weights[algo_name][all_seeds[0]], dataset_name, algo_name)#, x_axis=test_accuracies[algo_name])

    plot_values(train_epochs, train_accuracies, all_algos, 'Train accuracy', dataset_name)
    plot_values(train_epochs, train_losses, all_algos, 'log(Train loss)', dataset_name, log=True)
    plot_values(test_epochs, test_accuracies, all_algos, 'Test accuracy', dataset_name)
    plot_values(test_epochs, test_losses, all_algos, 'log(Test loss)', dataset_name, log=True)




