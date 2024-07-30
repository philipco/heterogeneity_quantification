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
    X = torch.concat([data_train, data_test])[:MAX_NB_POINTS]
    Y = torch.concat([labels_train, labels_test])[:MAX_NB_POINTS]

    print("Train data shape:", X[0].shape)
    print("Test data shape:", Y[0].shape)

    natural_split = False

    ### We generate a non-iid datasplit if it's not already done.
    X, Y = create_non_iid_split(X, Y, nb_of_clients, natural_split)

    train_features, test_features, train_labels, test_labels = train_test_split(X.float(), Y,
                                                                                test_size=len(labels_test),
                                                                                random_state=42)

    return X, Y, natural_split


def get_data_from_flamby(fed_dataset, nb_of_clients, kwargs_dataloader, debug: bool = False) \
        -> [List[torch.FloatTensor], List[torch.FloatTensor], bool]:

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

        X.append([data_train, data_test])
        Y.append([labels_train, labels_test])

    natural_split = True
    return X, Y, natural_split