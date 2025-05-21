"""
Purpose:
--------
This script benchmarks the convergence behavior of stochastic gradient descent (SGD) under three different training setups:
1. **Single Client Training**: A single model is trained using its local data.
2. **Multiple Clients with Gradient Averaging**: Each client computes gradients independently and they are averaged at
each iteration.
3. **Federated Averaging (FedAvg)**: Each client performs one or more local updates and then the models are averaged.

The goal is to have a simple framework where we can benchmark performance of local and federated training.

All clients work on a linear least squares regression task using synthetic data.
The goal is to empirically observe **variance reduction** in the optimization process due to averaging in multi-client settings,
especially in comparison to the single-client case.

Key Steps:
----------
- Generate synthetic datasets for each client using a linear model with small variations.
- Estimate the global Lipschitz constant L of the objective.
- Train models under three configurations and record the MSE loss over time.
- Plot the loss curves on a log scale to compare convergence rates and variance behavior.
"""

import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from huggingface_hub.utils import tqdm
from torch.utils.data import DataLoader

from src.data.DataLoader import generate_client_models
from src.data.DatasetConstants import NB_CLIENTS
from src.data.SyntheticDataset import SyntheticLSRDataset
from src.utils.UtilitiesPytorch import aggregate_gradients, aggregate_models, load_new_model
from src.optim.nn.Nets import Synth100ClientsRegression

torch.set_default_dtype(torch.float64)

# ---------- Training functions ----------

# SGD for a single client
def train_one_client(model, dataloader, L, steps=1000):
    criterion = nn.MSELoss()
    eta = 1 / (8 * L)  # Step size
    losses = []

    for i, (X, y) in enumerate(dataloader):
        if i >= steps:
            break
        y_pred = model(X)
        loss = criterion(y_pred, y)
        losses.append(loss.item())

        # Backpropagation and parameter update
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= eta * param.grad
                param.grad.zero_()

    return losses


# Synchronous gradient averaging across clients
def train_n_clients(models, dataloaders, L, steps=1000):
    eta = 1 / (8 * L)
    criterion = nn.MSELoss()
    losses = []

    for i, batches in enumerate(zip(*dataloaders)):
        if i >= steps:
            break
        total_loss = 0.0
        gradients = []

        # Compute gradients per client
        for model, (X, y) in zip(models, batches):
            y_pred = model(X)
            loss = criterion(y_pred, y)
            total_loss += loss.item()

            loss.backward()
            grad = [param.grad.detach() if param.grad is not None else None for param in model.parameters()]
            gradients.append(grad)

        # Average gradients across clients
        aggregated_gradients = aggregate_gradients(gradients, [1 / len(models)] * len(models), device="cpu")
        losses.append(total_loss / len(models))

        # Update all models with the averaged gradient
        with torch.no_grad():
            for model in models:
                for param, avg_p in zip(model.parameters(), aggregated_gradients):
                    param -= eta * avg_p
                    param.grad.zero_()

    return losses


# Federated Averaging training
def train_fedavg(models, dataloaders, L, global_steps=100, local_steps=1):
    eta = 1 / (8 * L)
    criterion = nn.MSELoss()
    losses = []

    for round_idx in tqdm(range(global_steps)):
        total_loss = 0.0

        # Local updates on each client
        for model, dataloader in zip(models, dataloaders):
            batch_iter = iter(dataloader)
            for _ in range(local_steps):
                X, y = next(batch_iter)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                loss.backward()

                gradients = [param.grad.detach() if param.grad is not None else None for param in model.parameters()]

                with torch.no_grad():
                    for param, avg_p in zip(model.parameters(), gradients):
                        param -= eta * avg_p
                        param.grad.zero_()

            total_loss += loss.item()

        # Global model averaging
        new_model = aggregate_models(models, [1 / len(models)] * len(models), device="cpu")
        for model in models:
            load_new_model(model, new_model)

        losses.append(total_loss / len(models))

    return losses


# ---------- Experiment setup ----------

# Parameters
d = 10  # Input dimensionality
batch_size = 1
num_clients = NB_CLIENTS["synth_complex"]
steps = 250  # Number of iterations

# Generate true model parameters for clients and their variations
true_theta, variations = generate_client_models(num_clients, 1, d, cluster_variance=0.01)

# Create datasets and estimate average Lipschitz constant
datasets = [SyntheticLSRDataset(m, v, batch_size) for (m, v) in zip(true_theta, variations)]
L = sum([d.L for d in datasets])/num_clients
print(f"Lipschitz constant L: {L}")

# ---------- Run training methods ----------

# Single client training
model_1 = Synth100ClientsRegression()
dataloader_1 = iter(DataLoader(datasets[0], batch_size=None))
losses_1 = train_one_client(model_1, dataloader_1, L, steps)

# Multi-client synchronous gradient averaging
net = Synth100ClientsRegression()
models_N = [copy.deepcopy(net) for _ in range(num_clients)]
dataloaders_N = [iter(DataLoader(d, batch_size=None)) for d in datasets]
losses_N = train_n_clients(models_N, dataloaders_N, L, steps)

# Multi-client FedAvg
net = Synth100ClientsRegression()
models_N = [copy.deepcopy(net) for _ in range(num_clients)]
dataloaders_N = [iter(DataLoader(d, batch_size=None)) for d in datasets]
losses_fed = train_fedavg(models_N, dataloaders_N, L, steps)

# ---------- Plot loss curves ----------

plt.plot(losses_1, label="1 Client")
plt.plot(losses_N, label=f"{num_clients} Clients (Averaged)")
plt.plot(losses_fed, label=f"{num_clients} Clients (FL)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("SGD Convergence: 1 Client vs N Clients Averaging")
plt.legend()
plt.yscale("log")
plt.show()
