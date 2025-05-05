import gc
from collections.abc import Sequence

import torch
from transformers import PreTrainedModel

from src.utils.UtilitiesPytorch import move_batch_to_device, assert_gradients_zero


def write_grad(trained_model, writer, last_epoch):
    # We plot the computed gradient on each client before their aggregation.
    for name, param in trained_model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'{name}.grad', param.grad, last_epoch)


def log_performance(name: str, net, device, loader, criterion, metric, client_ID, writer, epoch, optimal_loss, full_batch=False):
    # WARNING : For tcga_brca, we need to evaluate the metric on the full dataset.
    epoch_loss, epoch_accuracy = compute_loss_and_accuracy(net, device, loader, criterion, metric,
                                                           full_batch)
    # Writing logs.
    if optimal_loss:
        writer.add_scalar(f'{name}_loss', epoch_loss-optimal_loss, epoch)
        writer.add_scalar(f'{name}_accuracy', epoch_accuracy-optimal_loss, epoch)
    else:
        writer.add_scalar(f'{name}_loss', epoch_loss, epoch)
        writer.add_scalar(f'{name}_accuracy', epoch_accuracy, epoch)
    writer.close()
    return epoch_loss, epoch_accuracy


def train_local_neural_network(net, optimizer, scheduler, device, client_ID, train_loader, train_iter, criterion,
                               nb_local_epochs):
    """
    Train a neural network on a local dataset with a given optimizer, scheduler, and performance logging.

    This function trains a neural network using a provided training and validation dataset loader. It allows
    for optimizer and scheduler initialization if not provided, supports logging with TensorBoard, and tracks
    loss and accuracy throughout the training process.

    """
    train_loss = []

    # Training
    for local_epoch in range(nb_local_epochs):
        batch_training(train_loader, train_iter, device, net, criterion, optimizer)
    scheduler.step()
    return train_loss


def batch_training(train_loader, train_iter, device, net, criterion, optimizer):
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
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
    return gradient

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

def compute_loss_and_accuracy(net, device, data_loader, criterion, metric, full_batch = False):
    epoch_loss = 0
    epoch_accuracy = 0
    net.eval()
    if not full_batch:
        with torch.no_grad():
            if isinstance(data_loader, Sequence):
                features, labels = data_loader.dataset[:]
                x_batch = features.to(device)
                y_batch = labels.to(device)
                outputs = net(x_batch).float()
                epoch_loss = criterion(outputs, y_batch)
                epoch_accuracy = metric(y_batch, outputs)
            # FOR LLM
            elif isinstance(net, PreTrainedModel):
                batch = next(iter(data_loader))
                outputs = net(**move_batch_to_device(batch, device))
                epoch_loss += outputs.loss
                epoch_accuracy += metric(batch['labels'], outputs.logits)
                del batch
            else:
                for i in range(1):
                    x_batch, y_batch = next(iter(data_loader))
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    outputs = net(x_batch).float()
                    epoch_loss += criterion(outputs, y_batch)
                    epoch_accuracy += metric(y_batch, outputs)
                    del x_batch, y_batch
        return epoch_loss, epoch_accuracy
    with torch.no_grad():
        if isinstance(net, PreTrainedModel):
            for batch in data_loader:
                outputs = net(**move_batch_to_device(batch, device))
                epoch_loss += outputs.loss
                epoch_accuracy += metric(batch['labels'], outputs.logits)
                del batch
                return (epoch_loss / len(data_loader)).to("cpu"), (epoch_accuracy / len(data_loader)).to("cpu")
        else:
            if isinstance(data_loader, Sequence):
                for x_batch, y_batch in data_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    outputs = net(x_batch).float()
                    epoch_loss += criterion(outputs, y_batch)
                    epoch_accuracy += metric(y_batch, outputs)
                    del x_batch, y_batch
                    return (epoch_loss / len(data_loader)).to("cpu"), (epoch_accuracy / len(data_loader)).to("cpu")
            else:
                # For synthetic dataset that generates data on the fly.
                nb_pass_on_data = 20
                # model_shift = data_loader.dataset.true_theta - net.linear.weight.to("cpu")
                # loss = model_shift @ data_loader.dataset.covariance @ model_shift.T
                # print(loss)
                # return loss, loss  # + data_loader.dataset.noise_std**2

                for i in range(nb_pass_on_data):
                    x_batch, y_batch = next(iter(data_loader))
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    outputs = net(x_batch).float()
                    epoch_loss += criterion(outputs, y_batch)
                    epoch_accuracy += metric(y_batch, outputs)
                    del x_batch, y_batch
                return (epoch_loss / nb_pass_on_data).to("cpu"), (epoch_accuracy / nb_pass_on_data).to("cpu")

