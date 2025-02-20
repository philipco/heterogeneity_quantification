import torch
from torch.optim.lr_scheduler import LRScheduler

class LinearWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, plateau = 0, last_epoch=-1):
        """
        Implements a PyTorch-compatible linear learning rate schedule with warm-up.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to adjust learning rate for.
            warmup_steps (int): Number of steps for linear warm-up.
            total_steps (int): Total number of training steps.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.plateau = plateau

        # Ensure each parameter group has 'initial_lr'
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate at the current step."""
        step = self.last_epoch + 1  # PyTorch uses `last_epoch` tracking
        if step < self.warmup_steps:
            scale = step / self.warmup_steps  # Linear warm-up
        elif step < self.warmup_steps + self.plateau:
            scale = 1
        else:
            scale = max(0.0, (self.total_steps - step) / (self.total_steps - self.warmup_steps))  # Linear decay

        return [base_lr * scale for base_lr in self.base_lrs]
