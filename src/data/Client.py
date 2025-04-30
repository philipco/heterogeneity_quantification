import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR, StepLR, LambdaLR

from src.optim.LinearWarmupScheduler import LinearWarmupScheduler, ConstantLRScheduler
from src.optim.Train import train_local_neural_network, compute_loss_and_accuracy, \
    log_performance
from src.utils.LoggingWriter import LoggingWriter


class Client:

    def __init__(self, ID, tensorboard_dir: str, train_loader, val_loader, test_loader, net: nn.Module,
                 criterion, metric, step_size: int, momentum: int, weight_decay: int, batch_size: int,
                 scheduler_params: (int, int)):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {self.device}")

        self.ID = ID

        # Writer for TensorBoard
        self.writer = LoggingWriter(
            log_dir=f'/home/cphilipp/GITHUB/heterogeneity_quantification/runs/{tensorboard_dir}/{self.ID}')

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # The iterable dataset has no length (online setting, length is infinite).
        try:
            self.nb_train_points = len(self.train_loader.dataset)
        except TypeError:
            self.nb_train_points = 1

        self.trained_model = net.to(self.device)

        self.step_size, self.momentum, self.weight_decay = step_size, momentum, weight_decay
        self.optimizer = optim.SGD(net.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)
        self.set_step_scheduler(tensorboard_dir, scheduler_params[0], scheduler_params[1])
        if criterion is not None:
            self.criterion = criterion()
        else:
            self.criterion = None
        self.metric = metric
        self.last_epoch = 0
        self.optimal_loss = None

        # if "synth" in tensorboard_dir:
        #     with torch.no_grad():
        #         init_net = self.trained_model.linear.weight.clone()
        #         self.trained_model.linear.weight.copy_(self.train_loader.dataset.true_theta)
        #         self.optimal_loss, _ = compute_loss_and_accuracy(self.trained_model, self.device, self.train_loader, self.criterion, self.metric,
        #                                   full_batch = True)
        #         self.trained_model.linear.weight.copy_(init_net)

        self.writer.close()

    def set_step_scheduler(self, dataset_name, scheduler_steps, scheduler_gamma):
        if dataset_name in ["exam_llm"]:
            self.scheduler = LinearWarmupScheduler(self.optimizer, 5,
                                                   20, plateau=5)
        elif dataset_name in ["heart_disease"]:
            self.scheduler = StepLR(self.optimizer, step_size=scheduler_steps, gamma=scheduler_gamma)
        elif dataset_name in ["cifar10"]:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda t: self.step_size / (t + 1))
        else:
            self.scheduler = ConstantLRScheduler(self.optimizer)

    def reset_hyperparameters(self, net, step_size, momentum, weight_decay, scheduler_steps, scheduler_gamma,
                              dataset_name=None):
        self.last_epoch = 0
        self.trained_model = net.to(self.device)
        self.step_size, self.momentum = step_size, momentum
        self.optimizer = optim.SGD(net.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)
        self.set_step_scheduler(dataset_name, scheduler_steps, scheduler_gamma)

    def resplit_train_test(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test \
            = train_test_split(torch.concat([self.X_train, self.X_test]),
                               torch.concat([self.Y_train, self.Y_test]), test_size=self.test_size)

    def train(self, nb_epochs: int):
        # Compute train/val/test metrics at initialization
        self.write_train_val_test_performance()

        if nb_epochs != 0:
            self.train_loss \
                = train_local_neural_network(self.trained_model, self.optimizer, self.scheduler, self.device, self.ID,
                                             self.train_loader, self.val_loader, self.criterion, nb_epochs, self.step_size,
                                             self.momentum, self.metric, 0, 0)
            self.last_epoch += nb_epochs


            self.write_train_val_test_performance()

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

    def write_train_val_test_performance(self, logs="light"):
        if logs == "full":
            for name, param in self.trained_model.named_parameters():
                self.writer.add_histogram(f'{name}.weight', param, self.last_epoch)
        train_loss, train_acc = log_performance("train", self.trained_model, self.device, self.train_loader,
                                                self.criterion, self.metric, self.ID, self.writer, self.last_epoch,
                                                self.optimal_loss)
        log_performance("val", self.trained_model, self.device, self.val_loader, self.criterion, self.metric,
                        self.ID, self.writer, self.last_epoch, self.optimal_loss)
        test_loss, test_acc = log_performance("test", self.trained_model, self.device, self.test_loader,
                                              self.criterion, self.metric, self.ID, self.writer,
                                              self.last_epoch, self.optimal_loss, True)

        self.writer.add_scalar(f'generalisation_loss', abs(train_loss - test_loss), self.last_epoch)
        self.writer.add_scalar(f'generalisation_accuracy', abs(train_acc - test_acc), self.last_epoch)
        self.writer.close()

