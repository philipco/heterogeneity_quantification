"""Created by Constantin Philippenko, 5th April 2022."""

import os
from pathlib import Path
import random

import numpy as np
import psutil
import torch

def get_project_root() -> str:
    import pathlib
    path = str(pathlib.Path().absolute())
    root_dir = str(Path(__file__).parent.parent.parent)
    split = path.split(root_dir)
    return root_dir

def get_path_to_datasets() -> str:
    """Return the path to the datasets. For sake of anonymization, the path to datasets on clusters is not keep on
    GitHub and must be personalized locally"""
    return '{0}/../DATASETS/'.format(get_project_root())

def create_folder_if_not_existing(folder) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)


def file_exist(filename: str) -> None:
    return os.path.isfile(filename)


def remove_file(filename: str) -> None:
    os.remove(filename)


def print_mem_usage(info = None) -> None:
    if info:
        print(info)
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss * 10**-9 # Memory in Gb.
    print("Memory usage: {0:1.2f}Gb".format(mem))


def open_plotted_matrix(dataset_name: str, iid: bool = False) -> [np.ndarray, np.ndarray]:
    metrics_folder = "{0}/pictures/{1}".format(get_project_root(), dataset_name)
    type = "iid" if iid else "non_iid"
    non_iid_distance_X = np.loadtxt('{0}/{1}-{2}.txt'.format(metrics_folder, "X", type), delimiter=',')
    Y_name = "Y_TV" if OUTPUT_TYPE[dataset_name] == "discrete" else "Y"
    non_iid_distance_Y = np.loadtxt('{0}/{1}-{2}.txt'.format(metrics_folder, Y_name, type), delimiter=',')
    return non_iid_distance_X, non_iid_distance_Y

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False