"""Created by Constantin Philippenko, 29th September 2022."""
from typing import List

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.Dataset import prepare_liquid_asset
from src.data.Split import create_non_iid_split
from src.plot.PlotDifferentScenarios import f
from src.utils.Utilities import get_path_to_datasets


def get_dataloader(fed_dataset, train, kwargs_dataset, kwargs_dataloader):
    dataset = fed_dataset(train=train, **kwargs_dataset)
    return DataLoader(dataset, **kwargs_dataloader)


def get_element_from_dataloader(loader):
    if len(loader) == 0:
        return None, None
    X, Y = [], []
    for x, y in loader:
        X.append(x)
        Y.append(y)
    # For IXI, we should not flatten the dataset!
    # Same for tcga_brca.
    # TODO !
    return torch.concat(X), torch.concat(Y)#.flatten()


def get_data_from_csv(dataset_name: str) -> [List[torch.FloatTensor], List[torch.FloatTensor], bool]:

    root = get_path_to_datasets()
    if dataset_name == "liquid_asset":
        data_train, labels_train = prepare_liquid_asset(root, train=True)
    else:
        return ValueError("Dataset not recognized.")

    # Then for each (heterogeneous) client, we split the dataset into train/test
    X_train, X_val, X_test, Y_train, Y_val, Y_test = [], [], [], [], [], []

    for (x, y) in zip(data_train, labels_train):
        x2, x_test, y2, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=2024)
        X_train.append(x_train)
        X_val.append(x_val)
        X_test.append(x_test)
        Y_train.append(y_train)
        Y_val.append(y_val)
        Y_test.append(y_test)

    natural_split = True
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, natural_split


def get_synth_data(dataset_name: str) -> [List[torch.FloatTensor], List[torch.FloatTensor], bool]:

    n = 200
    X1, X2, Y1, Y2 = f("same_partionned_support", n=n)
    data_train = [torch.Tensor(X1).reshape(n,1), torch.Tensor(X2).reshape(n,1)]
    labels_train = [torch.Tensor(Y1).reshape(n, 1), torch.Tensor(Y2).reshape(n, 1)]
    # Then for each (heterogeneous) client, we split the dataset into train/test
    X_train, X_val, X_test, Y_train, Y_val, Y_test = [], [], [], [], [], []

    for (x, y) in zip(data_train, labels_train):
        x2, x_test, y2, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=2024)
        X_train.append(x_train)
        X_val.append(x_val)
        X_test.append(x_test)
        Y_train.append(y_train)
        Y_val.append(y_val)
        Y_test.append(y_test)

    natural_split = True
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, natural_split


def get_data_from_pytorch(fed_dataset, nb_of_clients, kwargs_train_dataset, kwargs_test_dataset,
                          kwargs_dataloader) -> [List[torch.FloatTensor], List[torch.FloatTensor], bool]:

    # Get dataloader for train/test.
    loader_train = get_dataloader(fed_dataset, train=True, kwargs_dataset=kwargs_train_dataset,
                            kwargs_dataloader=kwargs_dataloader)
    loader_test = get_dataloader(fed_dataset, train=False, kwargs_dataset=kwargs_test_dataset,
                            kwargs_dataloader=kwargs_dataloader)

    # Get all element from the dataloader.
    data_train, labels_train = get_element_from_dataloader(loader_train)
    data_test, labels_test = get_element_from_dataloader(loader_test)

    X = torch.concat([data_train, data_test])
    Y = torch.concat([labels_train, labels_test])

    print("Train data shape:", X[0].shape)
    print("Test data shape:", Y[0].shape)

    ### We generate a non-iid datasplit if it's not already done.
    X, Y = create_non_iid_split(X, Y, nb_of_clients, natural_split=False)

    # Then for each (heterogeneous) client, we split the dataset into train/test
    X_train, X_val, X_test, Y_train, Y_val, Y_test = [], [], [], [], [], []
    for (x,y) in zip(X, Y):
        x2, x_test, y2, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=2024)
        X_train.append(x_train)
        X_val.append(x_val)
        X_test.append(x_test)
        Y_train.append(y_train)
        Y_val.append(y_val)
        Y_test.append(y_test)

    natural_split = False
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, natural_split


def get_data_from_flamby(fed_dataset, nb_of_clients, dataset_name: str, kwargs_dataloader, debug: bool = False) \
        -> [List[torch.FloatTensor], List[torch.FloatTensor], bool]:

    X_train, X_test, X_val, Y_train, Y_val, Y_test = [], [], [], [], [], []
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

        # For TCGA_BRCA, there must be enough point to compute the metric, using the train set to create a very small
        # val set do not work.
        if dataset_name not in ["tcga_brca"]:
            data_train, data_val, labels_train, labels_val = train_test_split(data_train, labels_train,
                                                                              test_size=0.1, random_state=2023)
            X_val.append(torch.concat([data_val]))
            Y_val.append(torch.concat([labels_val]))
        else:
            X_val.append(torch.concat([data_test]))
            Y_val.append(torch.concat([labels_test]))

        X_train.append(torch.concat([data_train]))
        Y_train.append(torch.concat([labels_train]))
        X_test.append(torch.concat([data_test]))
        Y_test.append(torch.concat([labels_test]))

    natural_split = True
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, natural_split