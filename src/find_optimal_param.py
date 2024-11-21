"""Created by Constantin Philippenko, 29th September 2022."""

import optuna
from functools import partial
import argparse


from src.data.NetworkLoader import get_network
from src.optim.Algo import all_for_all_algo
from src.data.DatasetConstants import MODELS


# Define the training function
def objective(trial, network):
    # Define the hyperparameters
    step_size = trial.suggest_float("step_size", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_categorical("weight_decay", [0, 10**-4, 10**-3, 10**-2, 10**-1, 1])
    # scheduler_steps = trial.suggest_int("scheduler_steps", 1, 15)
    scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.5, 1)
    if network.dataset_name in ["heart_disease", "tcga_brca", "mnist"]:
        momentum = 0
    else:
        momentum = trial.suggest_categorical("momentum", [0, 0.9, 0.95, 0.99])
    net = MODELS[network.dataset_name]()
    for client in network.clients:
        client.reset_hyperparameters(net, step_size, momentum, weight_decay, 15, scheduler_gamma)
        network.trial = trial
    all_for_all_algo(network, nb_of_synchronization=25, pruning=True)
    return network.writer.get_scalar(f'test_accuracy', network.clients[0].last_epoch)

NB_TRIALS = 50

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset.",
        required=True,
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name

    assert dataset_name in ["mnist", "cifar10", "heart_disease", "tcga_brca", "ixi", "liquid_asset", "synth"], \
        "Dataset not recognized."
    print(f"### ================== DATASET: {dataset_name} ================== ###")

    network = get_network(dataset_name, algo_name="all_for_all", nb_initial_epochs=0)

    # Create a study object and optimize the objective function.
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

    study.optimize(partial(objective, network=network), n_trials=NB_TRIALS)

    # Retrieve the best configuration
    best_trial = study.best_trial
    print("Best trial config: ", best_trial.params)
    print("Best trial final validation accuracy: ", best_trial.value)




