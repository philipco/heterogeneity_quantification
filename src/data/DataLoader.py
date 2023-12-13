"""Created by Constantin Philippenko, 29th September 2022."""
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.Split import create_non_iid_split


def get_dataloader(fed_dataset, train, kwargs_dataset, kwargs_dataloader):
    dataset = fed_dataset(train=train, **kwargs_dataset)
    return DataLoader(dataset, **kwargs_dataloader)


def get_element_from_dataloader(loader):
    X, Y = [], []
    for x, y in loader:
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        X.append(x)
        Y.append(y)
    return np.concatenate(X), np.concatenate(Y)


def get_data_from_pytorch(fed_dataset, nb_of_clients, kwargs_dataset,
                          kwargs_dataloader) -> [List[np.array], List[np.array], bool]:

    # Get dataloader for train/test.
    loader_train = get_dataloader(fed_dataset, train=True, kwargs_dataset=kwargs_dataset,
                            kwargs_dataloader=kwargs_dataloader)
    loader_test = get_dataloader(fed_dataset, train=False, kwargs_dataset=kwargs_dataset,
                            kwargs_dataloader=kwargs_dataloader)

    # Get all element from the dataloader.
    data_train, labels_train = get_element_from_dataloader(loader_train)
    data_test, labels_test = get_element_from_dataloader(loader_test)
    X = [np.concatenate([data_train, data_test])]
    Y = [np.concatenate([labels_train, labels_test])]

    print("Train data shape:", X[0].shape)
    print("Test data shape:", Y[0].shape)

    natural_split = False
    X, Y = [np.concatenate(X)], [np.concatenate(Y)],

    ### We generate a non-iid datasplit if it's not already done.
    X, Y = create_non_iid_split(X, Y, nb_of_clients, natural_split)
    return X, Y, natural_split


def get_data_from_flamby(fed_dataset, nb_of_clients, kwargs_dataloader, debug: bool = False) \
        -> [List[np.array], List[np.array], bool]:

    X, Y = [], []
    for i in range(nb_of_clients):
        kwargs_dataset = dict(center=i, pooled=False)
        if debug:
            kwargs_dataset['debug'] = True
        loader_train = get_dataloader(fed_dataset, train=True, kwargs_dataset=kwargs_dataset,
                                kwargs_dataloader=kwargs_dataloader)

        loader_test = get_dataloader(fed_dataset, train=False, kwargs_dataset=kwargs_dataset,
                                kwargs_dataloader=kwargs_dataloader)

        # Get all element from the dataloader.
        data_train, labels_train = get_element_from_dataloader(loader_train)
        data_test, labels_test = get_element_from_dataloader(loader_test)

        X.append(np.concatenate([data_train, data_test]))
        Y.append(np.concatenate([labels_train, labels_test]))

    natural_split = True

    ### We generate a non-iid datasplit if it's not already done.
    X, Y = create_non_iid_split(X, Y, nb_of_clients, natural_split)
    return X, Y, natural_split