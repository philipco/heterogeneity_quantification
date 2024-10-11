import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.Utilities import set_seed


def log_performance(subset_name: str, net, device, loader, criterion, metric, client_ID, writer, epoch):
    # WARNING : For tcga_brca, we need to evaluate the metric on the full dataset.
    epoch_test_loss, epoch_test_accuracy = compute_loss_and_accuracy(net, device, loader,
                                                                     criterion, metric, "tcga_brca" in client_ID)
    # Writing logs.
    writer.add_scalar(f'{subset_name}_loss', epoch_test_loss, epoch)
    writer.add_scalar(f'{subset_name}_accuracy', epoch_test_accuracy, epoch)
    writer.close()


def train_local_neural_network(net, optimizer, scheduler, device, client_ID, train_loader, val_loader, criterion,
                               nb_epochs, lr, momentum, metric, last_epoch: int, writer: SummaryWriter, epoch):
    """Create train/test and train a neural network."""

    for name, param in net.named_parameters():
        writer.add_histogram(f'{name}.weight', param, 2 * epoch)

    # The optimizer should be initialized once at the beginning of the training.
    if optimizer is None:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    train_loss = []

    # Training
    print(f"=== Training the neural network on {client_ID}. ===")
    set_seed(last_epoch)
    for local_epoch in tqdm(range(nb_epochs)):
        net.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            net.zero_grad()

            # Forward pass
            outputs = net(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        scheduler.step()

        # We compute the train loss/performance-metric on the full train set after a full pass on it.
        # WARNING : For tcga_brca, we need to evaluate the metric on the full dataset.
        epoch_train_loss, epoch_train_accuracy = compute_loss_and_accuracy(net, device, train_loader, criterion, metric,
                                                                       "tcga_brca" in client_ID)
        train_loss.append(epoch_train_loss)

        # Writing logs.
        writer.add_scalar('train_loss', epoch_train_loss, local_epoch + last_epoch)
        writer.add_scalar('train_accuracy', epoch_train_accuracy, local_epoch + last_epoch)

        # We compute the val loss/performance-metric on the full train set after a full pass on it.


        log_performance("val", net, device, val_loader, criterion, metric, client_ID, writer,
                        local_epoch + last_epoch)

    for name, param in net.named_parameters():
        writer.add_histogram(f'{name}.weight', param, 2 * epoch + 1)

    # Close the writer at the end of training
    writer.close()

    print("Final train loss:", train_loss[-1])
    print(f"Final train accuracy: {epoch_train_accuracy}")

    return net, train_loss, writer, optimizer, scheduler


def compute_loss_and_accuracy(net, device, data_loader, criterion, metric, full_batch = False):
    epoch_loss = 0
    epoch_accuracy = 0
    net.eval()
    if full_batch:
        with torch.no_grad():
            features, labels = data_loader.dataset[:]
            x_batch = features.to(device)
            y_batch = labels.to(device)
            outputs = net(x_batch).float()
            epoch_loss = criterion(outputs, y_batch)
            epoch_accuracy = metric(y_batch, outputs)
        return epoch_loss, epoch_accuracy
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = net(x_batch).float()
            epoch_loss += criterion(outputs, y_batch)
            epoch_accuracy += metric(y_batch, outputs)
    return epoch_loss / len(data_loader), epoch_accuracy / len(data_loader)
