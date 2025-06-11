import copy

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, LambdaLR

from src.optim.LinearWarmupScheduler import LinearWarmupScheduler, ConstantLRScheduler
from src.optim.Train import log_performance, batch_training
from src.utils.LoggingWriter import LoggingWriter


class Client:
    """
    Represents a federated learning client participating in training.

    This class encapsulates the local training logic, optimizer and scheduler setup,
    and performance logging for an individual client in a federated learning setup.

    Attributes:
        ID (str): Unique identifier for the client.
        writer (LoggingWriter): TensorBoard writer for logging training and evaluation metrics.
        train_loader, val_loader, test_loader: Data loaders for training, validation, and testing datasets.
        trained_model (nn.Module): Neural network model to be trained locally.
        criterion (nn.Module): Loss function.
        metric (Callable): Evaluation metric (e.g., accuracy).
        optimizer (torch.optim.Optimizer): Optimizer for model updates.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        last_epoch (int): Tracks the current epoch for logging and scheduling.
    """

    def __init__(self, ID, tensorboard_dir: str, algo_name: str, initial_seed: int, train_loader, val_loader, test_loader, net: nn.Module,
                 criterion, metric, step_size: int, momentum: int, weight_decay: int, batch_size: int,
                 scheduler_params: (int, int)):
        super().__init__()

        # Device setup (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {self.device}")

        self.ID = ID

        # TensorBoard writer for tracking training logs
        self.writer = LoggingWriter(
            log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{tensorboard_dir}/{self.ID}')

        self.algo_name = algo_name
        self.initial_seed = initial_seed

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Determine training set size (1 if the dataset is a stream or doesn't have __len__)
        try:
            self.nb_train_points = len(self.train_loader.dataset)
        except TypeError:
            self.nb_train_points = 1

        self.trained_model = copy.deepcopy(net).to(self.device)
        if self.algo_name  in ["Ditto", "APFL"]:
            self.global_model = copy.deepcopy(net).to(self.device)
        if algo_name == "APFL":
            self.personalized_model = copy.deepcopy(net).to(self.device)

        self.optimizer = optim.SGD(self.trained_model.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)
        if self.algo_name  in ["Ditto", "APFL"]:
            self.global_optimizer = optim.SGD(self.global_model.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)
        if algo_name in ["APFL"]:
            self.personalized_optimizer = optim.SGD(self.personalized_model.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)

        # Learning rate scheduler
        self.step_size, self.momentum, self.weight_decay = step_size, momentum, weight_decay
        self.set_step_scheduler(tensorboard_dir, scheduler_params[0], scheduler_params[1])

        # Loss function (instantiated if not None)
        if criterion is not None:
            self.criterion = criterion()
        else:
            self.criterion = None

        self.metric = metric
        self.last_epoch = 0

        self.writer.close()

    def set_step_scheduler(self, dataset_name, scheduler_steps, scheduler_gamma):
        """
        Selects and sets the appropriate learning rate scheduler based on dataset name.
        """
        if dataset_name in ["LLM"]:
            self.scheduler = LinearWarmupScheduler(self.optimizer, 5, 20, plateau=5)
            if self.algo_name in ["Ditto", "APFL"]:
                self.global_scheduler = LinearWarmupScheduler(self.global_optimizer, 5, 20, plateau=5)
            if self.algo_name  in ["APFL"]:
                self.personalized_scheduler = LinearWarmupScheduler(self.personalized_optimizer, 5, 20, plateau=5)
        elif dataset_name in ["heart_disease", "mnist", "mnist_iid", "cifar10", "cifar10_iid", "ixi", "exam_llm"]:
            self.scheduler = StepLR(self.optimizer, step_size=scheduler_steps, gamma=scheduler_gamma)
            if self.algo_name in ["Ditto", "APFL"]:
                self.global_scheduler = StepLR(self.global_optimizer, step_size=scheduler_steps, gamma=scheduler_gamma)
            if self.algo_name in ["APFL"]:
                self.personalized_scheduler = StepLR(self.personalized_optimizer, step_size=scheduler_steps, gamma=scheduler_gamma)
        elif dataset_name in ["X"]:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda t: self.step_size / (t + 1))
            if self.algo_name in ["Ditto", "APFL"]:
                self.global_scheduler = LambdaLR(self.global_optimizer, lr_lambda=lambda t: self.step_size / (t + 1))
            if self.algo_name in ["APFL"]:
                self.personalized_scheduler = LambdaLR(self.personalized_optimizer, lr_lambda=lambda t: self.step_size / (t + 1))
        else:
            self.scheduler = ConstantLRScheduler(self.optimizer)
            if self.algo_name in ["Ditto", "APFL"]:
                self.global_scheduler = ConstantLRScheduler(self.global_optimizer)
            if self.algo_name in ["APFL"]:
                self.personalized_scheduler = ConstantLRScheduler(self.personalized_optimizer)

    def reset_hyperparameters(self, net, step_size, momentum, weight_decay, scheduler_steps, scheduler_gamma,
                              dataset_name=None):
        """
        Resets the model, optimizer, and scheduler with new hyperparameters.
        """
        self.last_epoch = 0
        self.trained_model = net.to(self.device)
        self.step_size, self.momentum = step_size, momentum
        self.optimizer = optim.SGD(net.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)
        self.set_step_scheduler(dataset_name, scheduler_steps, scheduler_gamma)

    def resplit_train_test(self):
        """
        Reshuffles and splits the training and test data (useful when data is merged dynamically).
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test \
            = train_test_split(torch.concat([self.X_train, self.X_test]),
                               torch.concat([self.Y_train, self.Y_test]), test_size=self.test_size)


    def write_train_val_test_performance(self, logs="light"):
        """
        Logs training and testing performance to TensorBoard.

        Args:
            logs (str): If "full", logs parameter histograms as well. If "light", logs only metrics.
        """
        if logs == "full":
            for name, param in self.trained_model.named_parameters():
                self.writer.add_histogram(f'{name}.weight', param, self.last_epoch)

        # Log training performance
        train_loss, train_acc = log_performance("train", self.trained_model, self.device, self.train_loader,
                                                self.criterion, self.metric, self.ID, self.writer, self.last_epoch,
                                                self.last_epoch == 0)

        # Log testing performance
        test_loss, test_acc = log_performance("test", self.trained_model, self.device, self.test_loader,
                                              self.criterion, self.metric, self.ID, self.writer,
                                              self.last_epoch, True)

        # Log generalization gap
        self.writer.add_scalar(f'generalisation_loss', abs(train_loss - test_loss), self.last_epoch)
        self.writer.add_scalar(f'generalisation_accuracy', abs(train_acc - test_acc), self.last_epoch)
        self.writer.close()
