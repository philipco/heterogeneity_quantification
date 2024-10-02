import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.optim.Train import train_local_neural_network, evaluate_test_metric


class Client:

    def __init__(self, ID, X_train, X_val, X_test, Y_train, Y_val, Y_test, output_dim: int, net: nn.Module,
                 criterion, metric, step_size: int, momentum: int, batch_size: int):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {self.device}")

        self.ID = ID

        # Writer for TensorBoard
        self.writer = SummaryWriter(
            log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{self.ID}_non_iid')

        self.train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size)
        self.val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

        self.X_test, self.Y_test = X_test.to(self.device), Y_test.to(self.device)

        self.input_dim = self.X_test.shape[1]
        self.output_dim = output_dim

        # Type of network to use, simply a class
        self.net = net
        self.optimizer = None
        self.criterion = criterion
        self.metric = metric
        self.step_size, self.momentum = step_size, momentum
        self.last_epoch = 0

        # self.projecteur = features @ torch.linalg.pinv(features.T @ features) @ features.T

    def resplit_train_test(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test \
            = train_test_split(torch.concat([self.X_train, self.X_test]),
                               torch.concat([self.Y_train, self.Y_test]), test_size=self.test_size)

    def train(self, nb_epochs: int, batch_size: int):
        criterion = self.criterion()

        self.trained_model, self.train_loss, self.writer, self.optimizer \
            = train_local_neural_network(self.net(), self.optimizer, self.device, self.ID, self.train_loader,
                                         self.val_loader, criterion, nb_epochs, self.step_size, self.momentum,
                                         self.metric, self.last_epoch, self.writer, 0)
        self.last_epoch += nb_epochs

        # Compute test metrics
        test_metric = evaluate_test_metric(self.trained_model, self.X_test, self.Y_test, self.metric)
        print(f"\nTest metric:", test_metric)
        # Compute test loss
        test_outputs = self.trained_model(self.X_test)
        self.test_loss = criterion(test_outputs, self.Y_test)
        atomic_criterion = self.criterion(reduction='none')
        self.atomic_test_losses = atomic_criterion(test_outputs, self.Y_test)

    def continue_training(self, nb_epochs: int, batch_size: int, epoch):
        criterion = self.criterion()

        self.trained_model, self.train_loss, self.writer, self.optimizer \
            = train_local_neural_network(self.trained_model, self.optimizer, self.device, self.ID, self.train_loader,
                                         self.val_loader, criterion, nb_epochs, self.step_size, self.momentum,
                                         self.metric, self.last_epoch, self.writer, epoch)

        torch.cuda.empty_cache()
        self.last_epoch += nb_epochs

        # Compute test metrics
        test_metric = evaluate_test_metric(self.trained_model, self.X_test, self.Y_test, self.metric)
        print(f"\nTest metric:", test_metric)
        # Compute test loss

        # Compute the test loss aggregated and atomic.
        atomic_criterion = self.criterion(reduction='none')
        self.test_loss, self.atomic_test_losses = [], []
        with torch.no_grad():
            for i in range(0, len(self.X_test), batch_size):
                x_batch = self.X_test[i:i + batch_size]  # .to(device)
                y_batch = self.Y_test[i:i + batch_size]  # .to(device)
                predictions = self.trained_model(x_batch)

                self.test_loss.append(criterion(predictions, y_batch))
                self.atomic_test_losses.append(atomic_criterion(predictions, y_batch))
        self.atomic_test_losses = torch.concat(self.atomic_test_losses)
        self.test_loss = torch.mean(torch.stack(self.test_loss))
        torch.cuda.empty_cache()
