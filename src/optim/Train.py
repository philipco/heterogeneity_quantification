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
    """
    Train a neural network on a local dataset with a given optimizer, scheduler, and performance logging.

    This function trains a neural network using a provided training and validation dataset loader. It allows
    for optimizer and scheduler initialization if not provided, supports logging with TensorBoard, and tracks
    loss and accuracy throughout the training process.

    :param net:
        The neural network model to train.
    :type net: torch.nn.Module

    :param optimizer:
        The optimizer used for training the model. If `None`, a new optimizer (SGD) will be created.
    :type optimizer: torch.optim.Optimizer or None

    :param scheduler:
        The learning rate scheduler used to adjust the learning rate during training. If `None`, a new scheduler
        (StepLR) will be created.
    :type scheduler: torch.optim.lr_scheduler or None

    :param device:
        The device to perform the computations on (e.g., 'cpu', 'cuda').
    :type device: str

    :param client_ID:
        Identifier for the client (useful for federated learning or distributed settings).
    :type client_ID: str

    :param train_loader:
        DataLoader for the training set, providing batches of training data.
    :type train_loader: torch.utils.data.DataLoader

    :param val_loader:
        DataLoader for the validation set, providing batches of validation data.
    :type val_loader: torch.utils.data.DataLoader

    :param criterion:
        The loss function used to compute the training loss.
    :type criterion: torch.nn.Module

    :param nb_epochs:
        Number of epochs to train the model.
    :type nb_epochs: int

    :param lr:
        Learning rate used for the optimizer (if it is initialized within the function).
    :type lr: float

    :param momentum:
        Momentum value used in the SGD optimizer (if it is initialized within the function).
    :type momentum: float

    :param metric:
        The performance metric to evaluate the model (e.g., accuracy).
    :type metric: callable

    :param last_epoch:
        The last completed epoch number, used for seeding and TensorBoard logging.
    :type last_epoch: int

    :param writer:
        TensorBoard SummaryWriter object for logging training statistics (loss, accuracy, histograms).
    :type writer: torch.utils.tensorboard.SummaryWriter

    :param epoch:
        Current global epoch used for logging histograms of model parameters.
    :type epoch: int

    :return:
        A tuple containing the trained model (`net`), list of training loss values, updated `writer`,
        `optimizer`, and `scheduler`.
    :rtype: Tuple[torch.nn.Module, List[float], torch.utils.tensorboard.SummaryWriter, torch.optim.Optimizer, torch.optim.lr_scheduler]

    **Example usage:**

    .. code-block:: python

        net = MyModel()
        optimizer = None
        scheduler = None
        device = 'cuda'
        train_loader = ...
        val_loader = ...
        criterion = torch.nn.CrossEntropyLoss()
        nb_epochs = 10
        lr = 0.01
        momentum = 0.9
        metric = accuracy_function
        last_epoch = 0
        writer = SummaryWriter()

        trained_model, train_loss, writer, optimizer, scheduler = train_local_neural_network(
            net, optimizer, scheduler, device, 'client_1', train_loader, val_loader,
            criterion, nb_epochs, lr, momentum, metric, last_epoch, writer, epoch=0
        )

    **Notes:**
    - This function uses the `torch.no_grad()` context to disable gradient tracking when logging model parameters.
    - It initializes the optimizer and scheduler if they are not provided.
    - The function writes training/validation statistics to TensorBoard for each epoch, including histograms of model weights.
    - The training process logs performance on both the training and validation sets after each epoch.
    """

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

        batch_training(train_loader, device, net, criterion, optimizer, scheduler)

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

def batch_training(train_loader, device, net, criterion, optimizer, scheduler):
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
