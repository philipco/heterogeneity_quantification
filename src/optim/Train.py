import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_neural_network(net, client_ID, X_train, X_val, Y_train, Y_val, criterion, nb_epochs, lr, momentum,
                         batch_size, metric):
    """Create train/test and train a neural network."""
    # Writer for TensorBoard
    writer = SummaryWriter(log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{client_ID}')

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
    train_loss = []

    # Training
    print(f"=== Training the neural network on {client_ID}. ===")
    for epoch in tqdm(range(nb_epochs)):
        net.train()
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i + batch_size]
            y_batch = Y_train[i:i + batch_size]

            net.zero_grad()

            # Forward pass
            outputs = net(x_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()

            for name, param in net.named_parameters():
                writer.add_histogram(f'{name}.grad', param.grad, epoch)
                writer.add_histogram(f'{name}.weight', param, epoch)

            optimizer.step()

        # We compute the train loss/performance-metric on the full train set after a full pass on it.
        epoch_train_loss = 0
        all_outputs = []
        epoch_accuracy = 0
        net.eval()
        with torch.no_grad():
            for i in range(0, len(X_train), batch_size):
                x_batch = X_train[i:i + batch_size]
                y_batch = Y_train[i:i + batch_size]

                outputs = net(x_batch).float()
                epoch_train_loss += criterion(outputs, y_batch)
                all_outputs.append(outputs)
        epoch_accuracy = metric(Y_train, torch.concat(all_outputs))
        epoch_train_loss /= len(Y_train)
        train_loss.append(epoch_train_loss)

        # Writing logs.
        writer.add_scalar('training_loss', epoch_train_loss, epoch)
        writer.add_scalar('training_accuracy', epoch_accuracy, epoch)

        scheduler.step(epoch_train_loss)

    # Close the writer at the end of training
    writer.close()

    print("Final train loss:", train_loss[-1])
    print("Final accuracy:", epoch_accuracy)

    return net, train_loss#, writer #, atomic_test_losses

def evaluate_test_metric(net, test_features, test_labels, metric):
    # Eval mode to deactivate any layers that behave differently during training.
    net.eval()
    with torch.no_grad():
        test_outputs = net(test_features)
        return metric(test_labels.detach().cpu(), test_outputs.detach().cpu())
