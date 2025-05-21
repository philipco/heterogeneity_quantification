"""
Utility functions for path management, file operations, memory usage monitoring, and reproducibility setup.
"""

import os
from pathlib import Path
import random

import numpy as np
import psutil
import torch


def get_project_root() -> str:
    """
    Return the absolute path to the root directory of the project.

    The root is assumed to be three levels up from the current file's location.
    """
    import pathlib
    path = str(pathlib.Path().absolute())
    root_dir = str(Path(__file__).parent.parent.parent)
    split = path.split(root_dir)
    return root_dir


def get_path_to_datasets() -> str:
    """
    Return the path to the datasets directory.

    This must be configured locally.
    """
    return '{0}/../DATASETS/'.format(get_project_root())


def create_folder_if_not_existing(folder) -> None:
    """
    Create the specified folder if it does not already exist.

    Args:
        folder (str): Path to the directory to be created.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def file_exist(filename: str) -> bool:
    """
    Check whether the specified file exists.

    Args:
        filename (str): Path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(filename)


def remove_file(filename: str) -> None:
    """
    Remove the specified file.

    Args:
        filename (str): Path to the file to be removed.
    """
    os.remove(filename)


def print_mem_usage(info=None) -> None:
    """
    Print the current memory usage of the process in gigabytes.

    Args:
        info (str, optional): Additional context to include in the output.
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss * 10**-9  # Memory in GB
    if info:
        print("{0}: {1:1.2f}Gb".format(info, mem))
    else:
        print("Memory usage: {0:1.2f}Gb".format(mem))


def set_seed(seed: int) -> None:
    """
    Set seeds for reproducibility across PyTorch, NumPy, and Python random.

    This also enforces deterministic behavior for CUDA operations.

    Args:
        seed (int): The seed to use.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
