import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_local_neural_network(net, device, client_ID, X_train, X_val, Y_train, Y_val, criterion, nb_epochs, lr, momentum,
                               batch_size, metric, last_epoch: int, writer: SummaryWriter = None):
    """Create train/test and train a neural network."""


    net.to(device)

    # Writer for TensorBoard
    if writer is None:
        writer = SummaryWriter(log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{client_ID}')

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    train_loss = []

    # Training
    print(f"=== Training the neural network on {client_ID}. ===")
    for epoch in tqdm(range(nb_epochs)):
        net.train()
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i + batch_size].to(device)
            y_batch = Y_train[i:i + batch_size].to(device)

            net.zero_grad()

            # Forward pass
            outputs = net(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()

            optimizer.step()

        # We compute the train loss/performance-metric on the full train set after a full pass on it.
        epoch_train_loss, epoch_train_accuracy = compute_loss_accuracy(net, device, X_train, Y_train, batch_size, criterion,
                                                                       metric)

        train_loss.append(epoch_train_loss)

        # Writing logs.
        writer.add_scalar('training_loss', epoch_train_loss, epoch + last_epoch)
        writer.add_scalar('training_accuracy', epoch_train_accuracy, epoch + last_epoch)

        # We compute the train loss/performance-metric on the full train set after a full pass on it.
        epoch_val_loss, epoch_val_accuracy = compute_loss_accuracy(net, device, X_val, Y_val, batch_size, criterion,
                                                                   metric)

        # Writing logs.
        writer.add_scalar('val_loss', epoch_val_loss, epoch + last_epoch)
        writer.add_scalar('val_accuracy', epoch_val_accuracy, epoch + last_epoch)

        # scheduler.step(epoch_train_loss)

    # Close the writer at the end of training
    writer.close()

    print("Final train loss:", train_loss[-1])
    print(f"Final train accuracy: {epoch_train_accuracy}\tFinal val accuracy: {epoch_val_accuracy}")

    return net, train_loss, writer


def compute_loss_accuracy(net, device, X, Y, batch_size, criterion, metric):
    epoch_loss = 0
    all_outputs = []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i + batch_size].to(device)
            y_batch = Y[i:i + batch_size].to(device)

            outputs = net(x_batch).float()
            epoch_loss += criterion(outputs, y_batch)
            all_outputs.append(outputs)
    epoch_accuracy = metric(Y.to(device), torch.concat(all_outputs))
    epoch_loss /= len(Y)
    return epoch_loss, epoch_accuracy

def evaluate_test_metric(net, test_features, test_labels, metric):
    # Eval mode to deactivate any layers that behave differently during training.
    net.eval()
    with torch.no_grad():
        test_outputs = net(test_features)
        return metric(test_labels.detach().cpu(), test_outputs.detach().cpu())
