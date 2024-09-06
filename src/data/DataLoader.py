"""Created by Constantin Philippenko, 29th September 2022."""
from typing import List

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.DatasetConstants import MAX_NB_POINTS
from src.data.Split import create_non_iid_split


def get_dataloader(fed_dataset, train, kwargs_dataset, kwargs_dataloader):
    dataset = fed_dataset(train=train, **kwargs_dataset)
    return DataLoader(dataset, **kwargs_dataloader)


def get_element_from_dataloader(loader):
    X, Y = [], []
    for x, y in loader:
        X.append(x)
        Y.append(y)
    # For IXI, we should not flatten the dataset!
    # Same for tcga_brca.
    # TODO !
    return torch.concat(X), torch.concat(Y)#.flatten()


def get_data_from_pytorch(fed_dataset, nb_of_clients, kwargs_dataset,
                          kwargs_dataloader) -> [List[torch.FloatTensor], List[torch.FloatTensor], bool]:

    # Get dataloader for train/test.
    loader_train = get_dataloader(fed_dataset, train=True, kwargs_dataset=kwargs_dataset,
                            kwargs_dataloader=kwargs_dataloader)
    loader_test = get_dataloader(fed_dataset, train=False, kwargs_dataset=kwargs_dataset,
                            kwargs_dataloader=kwargs_dataloader)

    # Get all element from the dataloader.
    data_train, labels_train = get_element_from_dataloader(loader_train)
    data_test, labels_test = get_element_from_dataloader(loader_test)
    # TODO : remove MAX_NB_POINTS.
    X = torch.concat([data_train, data_test])[:MAX_NB_POINTS]
    Y = torch.concat([labels_train, labels_test])[:MAX_NB_POINTS]

    print("Train data shape:", X[0].shape)
    print("Test data shape:", Y[0].shape)

    natural_split = False

    ### We generate a non-iid datasplit if it's not already done.
    X, Y = create_non_iid_split(X, Y, nb_of_clients, natural_split)

    # Then for each (heterogeneous) client, we split the dataset into train/test
    X_train, X_test, Y_train, Y_test = [], [], [], []
    for (x,y) in zip(X, Y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        X_train.append(x_train)
        X_test.append(x_test)
        Y_train.append(y_train)
        Y_test.append(y_test)

    return X_train, X_test, Y_train, Y_test, natural_split


def get_data_from_flamby(fed_dataset, nb_of_clients, kwargs_dataloader, debug: bool = False) \
        -> [List[torch.FloatTensor], List[torch.FloatTensor], bool]:

    X_train, X_test, Y_train, Y_test = [], [], [], []
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

        X_train.append(torch.concat([data_train]))
        Y_train.append(torch.concat([labels_train]))
        X_test.append(torch.concat([data_test]))
        Y_test.append(torch.concat([labels_test]))

    natural_split = True
    return X_train, X_test, Y_train, Y_test, natural_split