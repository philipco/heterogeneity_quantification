import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from huggingface_hub.utils import tqdm
from torch.utils.data import DataLoader

from src.data.DataLoader import generate_client_models
from src.data.DatasetConstants import NB_CLIENTS
from src.data.SyntheticDataset import SyntheticLSRDataset
from src.optim.PytorchUtilities import aggregate_gradients, aggregate_models, load_new_model
from src.optim.nn.Nets import Synth100ClientsRegression


# Training function for one client
def train_one_client(model, dataloader, L, steps=1000):
    criterion = nn.MSELoss()
    eta = 1 / (2 * L)  # Step size
    losses = []

    for i, (X, y) in enumerate(dataloader):
        if i >= steps:
            break
        y_pred = model(X)
        loss = criterion(y_pred, y)
        losses.append(loss.item())

        # Compute gradient and update weights
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= eta * param.grad
                param.grad.zero_()

    return losses

# Training function for N clients with gradient averaging
def train_n_clients(models, dataloaders, L, steps=1000):
    eta = 1 / (2 * L)  # Step size
    criterion = nn.MSELoss()
    losses = []

    for i, batches in enumerate(zip(*dataloaders)):
        if i >= steps:
            break
        total_loss = 0.0

        gradients = []

        # Compute gradients for each client
        for model, (X, y) in zip(models, batches):
            y_pred = model(X)
            loss = criterion(y_pred, y)
            total_loss += loss.item()

            loss.backward()
            grad = [param.grad.detach() if (param.grad is not None) else None for param in model.parameters()]
            gradients.append(grad)

        aggregated_gradients = aggregate_gradients(gradients, [1 / len(models) for i in range(len(models))], device="cpu")
        losses.append(total_loss / len(models))

        # Apply averaged gradient to all models
        with torch.no_grad():
            for model in models:
                for param, avg_p in zip(model.parameters(), aggregated_gradients):
                    param -= eta * avg_p
                    param.grad.zero_()

    return losses

# Federated Averaging Training Function
def train_fedavg(models, dataloaders, L, global_steps=100, local_steps=1):
    eta = 1 / (2 * L)  # Step size (used in local updates)
    criterion = nn.MSELoss()
    losses = []

    for round_idx in tqdm(range(global_steps)):

        total_loss = 0.0
        # Local training on each client
        for model, dataloader in zip(models, dataloaders):

            batch_iter = iter(dataloader)
            for _ in range(local_steps):
                X, y = next(batch_iter)

                y_pred = model(X)
                loss = criterion(y_pred, y)
                loss.backward()

                gradients = [param.grad.detach() if (param.grad is not None) else None for param in model.parameters()]

                with torch.no_grad():
                    for param, avg_p in zip(model.parameters(), gradients):
                        param -= eta * avg_p
                        param.grad.zero_()

            total_loss += loss.item()

        # Average model weights
        new_model = aggregate_models(models, [1 / len(models) for i in range(len(models))], device="cpu")
        for model in models:
            load_new_model(model, new_model)
        losses.append(total_loss / len(models))

    return losses

# Initialize experiment
d = 10  # Input dimension  # Ground truth parameter
batch_size = 1
num_clients = NB_CLIENTS["synth_complex"]

true_theta, variations = generate_client_models(num_clients, 1, d, cluster_variance=0.01)
steps = 250

# Create datasets and compute Lipschitz constant
datasets = [SyntheticLSRDataset(m, v, batch_size) for (m, v) in zip(true_theta, variations)]
L = sum([d.L for d in datasets])/num_clients
print(f"Lipschitz constant L: {L}")

# One-client case
model_1 = Synth100ClientsRegression()
dataloader_1 = iter(DataLoader(datasets[0], batch_size=None))
losses_1 = train_one_client(model_1, dataloader_1, L, steps)

# Multi-client case
net = Synth100ClientsRegression()
models_N = [copy.deepcopy(net) for _ in range(num_clients)]
dataloaders_N = [iter(DataLoader(d, batch_size=None)) for d in datasets]
losses_N = train_n_clients(models_N, dataloaders_N, L, steps)

# Multi-client case with FL
net = Synth100ClientsRegression()
models_N = [copy.deepcopy(net) for _ in range(num_clients)]
dataloaders_N = [iter(DataLoader(d, batch_size=None)) for d in datasets]
losses_fed = train_fedavg(models_N, dataloaders_N, L, steps)

# Plot results
plt.plot(losses_1, label="1 Client")
plt.plot(losses_N, label=f"{num_clients} Clients (Averaged)")
plt.plot(losses_fed, label=f"{num_clients} Clients (FL)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("SGD Convergence: 1 Client vs N Clients Averaging")
plt.legend()
plt.yscale("log")
plt.show()
