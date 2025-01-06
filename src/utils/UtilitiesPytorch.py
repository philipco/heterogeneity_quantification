import torch


def move_batch_to_device(batch, device):
    """
    Move a batch of data (a dictionary of tensors) to the specified device.

    Args:
        batch (dict): A dictionary where the values are tensors.
        device (torch.device): The device to move the tensors to (e.g., 'cpu' or 'cuda').

    Returns:
        dict: The batch with all tensors moved to the specified device.
    """
    return {k: v.to(device) for k, v in batch.items()}


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)