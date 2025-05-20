"""Created by Constantin Philippenko, 29th September 2022."""
import numpy as np
import optuna
from functools import partial
import argparse

from transformers import AutoModelForMultipleChoice

from src.data.Network import get_network
from src.optim.Algo import all_for_one_algo
from src.data.DatasetConstants import MODELS, CHECKPOINT


# Define the training function
def objective(trial, network):
    # Define the hyperparameters
    step_size = trial.suggest_float("step_size", 1e-4, 1, log=True)
    weight_decay = trial.suggest_categorical("weight_decay", [0, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1])
    scheduler_gamma = trial.suggest_float("scheduler_gamma", 0.1, 1)
    if network.dataset_name in ["synth_complex", "synth", "heart_disease", "tcga_brca", "mnist", "liquid_asset"]:
        momentum = 0
    else:
        momentum = 0.95
    scheduler_step = 50 if network.dataset_name in ["tcga_brca"] else trial.suggest_int("scheduler_steps", 1, 10)
        # momentum = trial.suggest_categorical("momentum", [0, 0.9, 0.95, 0.99])
    if network.dataset_name in ["exam_llm"]:
        net = AutoModelForMultipleChoice.from_pretrained(CHECKPOINT, cache_dir="./")
        # Freeze all pretrained weights
        for param in net.base_model.parameters():
            param.requires_grad = False
    else:
        net = MODELS[network.dataset_name]()
    for client in network.clients:
        client.reset_hyperparameters(net, step_size, momentum, weight_decay, scheduler_step, scheduler_gamma, network.dataset_name)#, scheduler_step, scheduler_gamma)
        network.trial = trial
    all_for_one_algo(network, nb_of_synchronization=15, pruning=True, continuous=True)
    scale = -1 if "synth" in dataset_name else 1 # For synthetic datasets we use the loss instead of the accuracy.
    ### We want to minimize the loss of the last 5 iterate (to reduce oscillation if case of).
    return scale * np.mean([network.writer.get_scalar(f'val_accuracy', network.clients[0].last_epoch - i).to("cpu") for i in range(5)])

NB_TRIALS = 10

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

    assert dataset_name in ["mnist", "cifar10", "heart_disease", "tcga_brca", "ixi", "liquid_asset", "synth", "synth_complex",
                            "exam_llm"], "Dataset not recognized."
    print(f"### ================== DATASET: {dataset_name} ================== ###")

    network = get_network(dataset_name, algo_name="All-for-one-cont")

    # Create a study object and optimize the objective function.
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())

    study.optimize(partial(objective, network=network), n_trials=NB_TRIALS)

    # Retrieve the best configuration
    best_trial = study.best_trial
    print("Best trial config: ", best_trial.params)
    print("Best trial final validation accuracy: ", best_trial.value)




