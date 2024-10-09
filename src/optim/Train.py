import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.Utilities import set_seed


def train_local_neural_network(net, optimizer, scheduler, device, client_ID, train_loader, val_loader, criterion, nb_epochs, lr, momentum,
                               metric, last_epoch: int, writer: SummaryWriter, epoch):
    """Create train/test and train a neural network."""

    for name, param in net.named_parameters():
        writer.add_histogram(f'{name}.weight', param, 2 * epoch)

    # The optimizer should be initialized once at the beginning of the training.
    if optimizer is None:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

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
        epoch_train_loss, epoch_train_accuracy = compute_loss_accuracy(net, device, train_loader, criterion, metric)
        train_loss.append(epoch_train_loss)

        # Writing logs.
        writer.add_scalar('training_loss', epoch_train_loss, local_epoch + last_epoch)
        writer.add_scalar('training_accuracy', epoch_train_accuracy, local_epoch + last_epoch)

        # We compute the train loss/performance-metric on the full train set after a full pass on it.
        epoch_val_loss, epoch_val_accuracy = compute_loss_accuracy(net, device, val_loader, criterion, metric)

        # Writing logs.
        writer.add_scalar('val_loss', epoch_val_loss, local_epoch + last_epoch)
        writer.add_scalar('val_accuracy', epoch_val_accuracy, local_epoch + last_epoch)

    for name, param in net.named_parameters():
        writer.add_histogram(f'{name}.weight', param, 2 * epoch + 1)

    # Close the writer at the end of training
    writer.close()

    print("Final train loss:", train_loss[-1])
    print(f"Final train accuracy: {epoch_train_accuracy}\tFinal val accuracy: {epoch_val_accuracy}")

    return net, train_loss, writer, optimizer, scheduler


def compute_loss_accuracy(net, device, data_loader, criterion, metric):
    epoch_loss = 0
    epoch_accuracy = 0
    net.eval()
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = net(x_batch).float()
            epoch_loss += criterion(outputs, y_batch)
            epoch_accuracy += metric(y_batch, outputs)
    return epoch_loss / len(data_loader), epoch_accuracy / len(data_loader)


def evaluate_test_metric(net, test_features, test_labels, metric):
    # Eval mode to deactivate any layers that behave differently during training.
    net.eval()
    with torch.no_grad():
        test_outputs = net(test_features)
        return metric(test_labels.detach().cpu(), test_outputs.detach().cpu())
