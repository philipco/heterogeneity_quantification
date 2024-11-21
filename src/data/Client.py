import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader, TensorDataset

from src.optim.Train import train_local_neural_network, write_train_val_test_performance
from src.utils.LoggingWriter import LoggingWriter


class Client:

    def __init__(self, ID, tensorboard_dir: str, X_train, X_val, X_test, Y_train, Y_val, Y_test, net: nn.Module,
                 criterion, metric, step_size: int, momentum: int, batch_size: int, scheduler_params: (int, int)):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {self.device}")

        self.ID = ID

        # Writer for TensorBoard
        self.writer = LoggingWriter(
            log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{tensorboard_dir}/{self.ID}')

        self.train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size)
        self.val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)
        self.test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size)

        self.trained_model = net.to(self.device)

        self.step_size, self.momentum = step_size, momentum
        self.optimizer = optim.SGD(net.parameters(), lr=step_size, momentum=momentum)
        self.scheduler = ConstantLR(self.optimizer, step_size=scheduler_params[0], gamma=scheduler_params[1])
        self.criterion = criterion()
        self.metric = metric
        self.last_epoch = 0
        self.writer.close()

    def reset_hyperparameters(self, net, step_size, momentum, weight_decay, scheduler_steps, scheduler_gamma):
        self.last_epoch = 0
        self.trained_model = net.to(self.device)
        self.step_size, self.momentum = step_size, momentum
        self.optimizer = optim.SGD(net.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = ConstantLR(self.optimizer, total_iters=scheduler_steps, factor=scheduler_gamma)

    def resplit_train_test(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test \
            = train_test_split(torch.concat([self.X_train, self.X_test]),
                               torch.concat([self.Y_train, self.Y_test]), test_size=self.test_size)

    def train(self, nb_epochs: int):
        # Compute train/val/test metrics at initialization
        write_train_val_test_performance(self.trained_model, self.device, self.train_loader,
                                         self.val_loader, self.test_loader, self.criterion, self.metric,
                                         self.ID, self.writer, self.last_epoch)

        self.train_loss \
            = train_local_neural_network(self.trained_model, self.optimizer, self.scheduler, self.device, self.ID,
                                         self.train_loader, self.val_loader, self.criterion, nb_epochs, self.step_size,
                                         self.momentum, self.metric, 0, 0)
        self.last_epoch += nb_epochs

        write_train_val_test_performance(self.trained_model, self.device, self.train_loader,
                                         self.val_loader, self.test_loader, self.criterion, self.metric,
                                         self.ID, self.writer, self.last_epoch)

    def continue_training(self, nb_of_local_epoch: int, current_epoch: int, single_batch: bool = False):
        self.train_loss \
            = train_local_neural_network(self.trained_model, self.optimizer, self.scheduler, self.device, self.ID,
                                         self.train_loader, self.val_loader, self.criterion, nb_of_local_epoch,
                                         self.step_size, self.momentum, self.metric, self.last_epoch, current_epoch,
                                         single_batch)

        # In the case of single batch training, we start iterating on the dataset from 0 and then use a modulo to iterate
        # thought the complete set.
        self.last_epoch += nb_of_local_epoch
        torch.cuda.empty_cache()

