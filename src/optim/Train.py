import gc
from collections.abc import Sequence
from itertools import islice

import torch
from transformers import PreTrainedModel

from src.data.SyntheticDataset import SyntheticLSRDataset
from src.utils.UtilitiesPytorch import move_batch_to_device, assert_gradients_zero


def write_grad(trained_model, writer, last_epoch):
    # We plot the computed gradient on each client before their aggregation.
    for name, param in trained_model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'{name}.grad', param.grad, last_epoch)


def log_performance(name: str, net, device, loader, criterion, metric, client_ID, writer, epoch, full_batch=False):
    # WARNING : For tcga_brca, we need to evaluate the metric on the full dataset.
    epoch_loss, epoch_accuracy = compute_loss_and_accuracy(net, device, loader, criterion, metric,
                                                           full_batch)
    # Writing logs.
    writer.add_scalar(f'{name}_loss', epoch_loss, epoch)
    writer.add_scalar(f'{name}_accuracy', epoch_accuracy, epoch)
    writer.close()
    return epoch_loss, epoch_accuracy


def batch_training(train_iter, device, net, criterion, optimizer):
    batch = next(train_iter)
    net.zero_grad()
    if isinstance(net, PreTrainedModel):
        outputs = net(**move_batch_to_device(batch, device))
        loss = outputs.loss
    else:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Forward pass
        outputs = net(x_batch)
        loss = criterion(outputs, y_batch)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

def compute_gradient_validation_set(val_loader, net, device, optimizer, criterion):
    net.train()
    optimizer.zero_grad()
    assert_gradients_zero(net)

    # HuggingFace datasets
    accumulated_grads = [torch.zeros_like(param) if param.grad is not None else None for param in net.parameters()]
    for batch in val_loader:
        if isinstance(net, PreTrainedModel):
            outputs = net(**move_batch_to_device(batch, device))
            loss = outputs.loss

            del batch

        # Pytorch datasets
        else:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = net(x_batch)
            loss = criterion(outputs, y_batch)

            del x_batch, y_batch

        # Backward pass
        loss.backward()

        for i, param in enumerate(net.parameters()):
            if param.grad is not None:
                if accumulated_grads[i] is None:
                    accumulated_grads[i] = param.grad.clone().detach()
                else:
                    accumulated_grads[i] += param.grad.clone().detach()

        torch.cuda.empty_cache()
        gc.collect()

    return [g / len(val_loader) for g in accumulated_grads if g is not None]

def safe_gradient_computation(train_loader, iter_loader, device, trained_model, criterion, optimizer, scheduler):
    try:
        gradient = gradient_step(iter_loader, device, trained_model, criterion, optimizer, scheduler)
    except StopIteration:
        iter_loader = iter(train_loader)
        gradient = gradient_step(iter_loader, device, trained_model, criterion, optimizer, scheduler)
    return gradient, iter_loader

def gradient_step(train_iter, device, net, criterion, optimizer, scheduler):
    net.train()

    optimizer.zero_grad()
    assert_gradients_zero(net)

    # HuggingFace datasets
    if isinstance(net, PreTrainedModel):
        batch = next(train_iter)
        outputs = net(**move_batch_to_device(batch, device))
        loss = outputs.loss
        loss.backward()
        del batch
    # Pytorch datasets
    else:
        x_batch, y_batch = next(train_iter)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = net(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        del x_batch, y_batch

    torch.cuda.empty_cache()

    return [param.grad.detach() if (param.grad is not None) else None for param in net.parameters()]


def update_model(net, aggregated_gradients, optimizer):
    ### Check that I am not updating with the preivous grad.
    """
    Updates the model using the aggregated gradients.

    net: the neural network model to update
    aggregated_gradients: list of aggregated gradients to apply
    optimizer: the optimizer used for updating the model
    """
    net.train()
    optimizer.zero_grad()  # Clears any lingering gradients
    assert_gradients_zero(net)

    for param, grad in zip(net.parameters(), aggregated_gradients):
        param.grad = grad

    optimizer.step()

def compute_loss_and_accuracy(net, device, data_loader, criterion, metric, full_batch=False, nb_eval_batches=5):
    """
    Dispatch evaluation based on model and data_loader type.

    Handles four cases:
    1. HuggingFace model (PreTrainedModel)
    2. Sequence-based dataset (e.g., list or indexable dataset)
    3. Standard PyTorch DataLoader
    4. Synthetic iterable datasets (no __len__, infinite yield)

    Parameters:
    - net: The model to evaluate (can be a standard PyTorch model or a HuggingFace transformer).
    - device: The torch.device to run evaluation on (e.g., 'cpu' or 'cuda').
    - data_loader: DataLoader, custom iterable, or Sequence for evaluation.
    - criterion: Loss function (e.g., torch.nn.CrossEntropyLoss).
    - metric: Accuracy or other evaluation metric function.
    - full_batch (bool): If True, evaluates on the whole dataset; otherwise, uses nb_eval_batches batches.
    - nb_eval_batches (int): Number of batches to evaluate if full_batch is False.

    Returns:
    - Tuple of (mean_loss, mean_accuracy)
    """
    net.eval()
    with torch.no_grad():
        if isinstance(net, PreTrainedModel):
            return _eval_hf_model(net, device, data_loader, criterion, metric, full_batch, nb_eval_batches)
        elif isinstance(data_loader, Sequence):
            return _eval_sequence_model(net, device, data_loader, criterion, metric, full_batch, nb_eval_batches)
        elif isinstance(getattr(data_loader, "dataset", None), SyntheticLSRDataset):
            return _eval_iterable_model(net, device, data_loader, criterion, metric, full_batch, nb_eval_batches)
        else:
            return _eval_standard_model(net, device, data_loader, criterion, metric, full_batch, nb_eval_batches)

def _eval_standard_model(net, device, data_loader, criterion, metric, full_batch=False, nb_eval_batches=5):
    """
    Evaluation for standard PyTorch models using DataLoader.
    Assumes batches are in (features, labels) format.
    """
    epoch_loss = 0
    epoch_accuracy = 0

    if full_batch:
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass and convert to float if needed for metric compatibility
            outputs = net(x_batch).float()
            epoch_loss += criterion(outputs, y_batch)
            epoch_accuracy += metric(y_batch, outputs)

            # Manual cleanup to release GPU memory
            del x_batch, y_batch

        # Return average metrics (detached from GPU for safe logging)
        return (epoch_loss / len(data_loader)).to("cpu"), (epoch_accuracy / len(data_loader)).to("cpu")
    else:
        # Partial evaluation for faster debugging or low-resource setups
        data_iter = iter(data_loader)
        for x_batch, y_batch in islice(data_iter, nb_eval_batches):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = net(x_batch).float()
            epoch_loss += criterion(outputs, y_batch)
            epoch_accuracy += metric(y_batch, outputs)
            del x_batch, y_batch
        return epoch_loss / nb_eval_batches, epoch_accuracy / nb_eval_batches

def _eval_sequence_model(net, device, data_loader, criterion, metric, full_batch=False, nb_eval_batches=5):
    """
    Evaluation function for datasets provided as raw Sequences (not torch.utils.data.DataLoader).
    Assumes the data_loader has a `.dataset[:]` interface returning full features and labels.
    """
    epoch_loss = 0
    epoch_accuracy = 0

    if full_batch:
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = net(x_batch).float()
            epoch_loss += criterion(outputs, y_batch)
            epoch_accuracy += metric(y_batch, outputs)
            del x_batch, y_batch

        return (epoch_loss / len(data_loader)).to("cpu"), (epoch_accuracy / len(data_loader)).to("cpu")
    else:
        # NOTE: This assumes that `dataset[:]` returns the entire dataset as tensors
        features, labels = data_loader.dataset[:]
        x_batch = features.to(device)
        y_batch = labels.to(device)
        outputs = net(x_batch).float()
        epoch_loss = criterion(outputs, y_batch)
        epoch_accuracy = metric(y_batch, outputs)
        return epoch_loss, epoch_accuracy

def _eval_hf_model(net, device, data_loader, criterion, metric, full_batch=False, nb_eval_batches=5):
    """
    Evaluation for HuggingFace models that return an object with `.loss` and `.logits`.
    Each batch must be a dictionary with keys like 'input_ids', 'attention_mask', 'labels', etc.
    """
    epoch_loss = 0
    epoch_accuracy = 0

    if full_batch:
        for batch in data_loader:
            # Move each tensor in the batch dict to the target device
            inputs = move_batch_to_device(batch, device)
            outputs = net(**inputs)  # outputs: a ModelOutput object
            epoch_loss += outputs.loss
            epoch_accuracy += metric(batch['labels'], outputs.logits)
            del batch
        return (epoch_loss / len(data_loader)).to("cpu"), (epoch_accuracy / len(data_loader)).to("cpu")
    else:
        data_iter = iter(data_loader)
        for batch in islice(data_iter, nb_eval_batches):
            inputs = move_batch_to_device(batch, device)
            outputs = net(**inputs)
            epoch_loss += outputs.loss
            epoch_accuracy += metric(batch['labels'], outputs.logits)
            del batch
        return epoch_loss / nb_eval_batches, epoch_accuracy / nb_eval_batches

def _eval_iterable_model(net, device, data_loader, criterion, metric, full_batch=False, nb_eval_batches=5):
    """
    Evaluation function for infinite iterable datasets (e.g., synthetic data generated on-the-fly).
    """
    epoch_loss = 0
    epoch_accuracy = 0

    if full_batch:
        nb_eval_batches = 25
    data_iter = iter(data_loader)
    for _ in range(nb_eval_batches):
        x_batch, y_batch = next(data_iter)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = net(x_batch).float()
        epoch_loss += criterion(outputs, y_batch)
        epoch_accuracy += metric(y_batch, outputs)

        del x_batch, y_batch

    return epoch_loss / nb_eval_batches, epoch_accuracy / nb_eval_batches
