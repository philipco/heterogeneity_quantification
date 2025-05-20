"""Data loading and preprocessing utilities for federated learning experiments.

This module provides functions to load and preprocess datasets from various sources,
including CSV files, PyTorch datasets, FLamby datasets, and NLP datasets. It also
includes utilities for generating synthetic datasets and client models.
"""

import gc
from typing import List, Any

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from src.data.DataCollatorForMultipleChoice import DataCollatorForMultipleChoice
from src.data.LiquidAssetDataset import prepare_liquid_asset
from src.data.DatasetConstants import CHECKPOINT
from src.data.Split import create_non_iid_split
from src.data.SyntheticDataset import SyntheticLSRDataset
from src.utils.Utilities import get_path_to_datasets, print_mem_usage


def get_dataloader(fed_dataset, train, kwargs_dataset, kwargs_dataloader):
    """Create a DataLoader for the specified federated dataset.

    Args:
        fed_dataset: The federated dataset class.
        train: Boolean indicating whether to load the training split.
        kwargs_dataset: Dictionary of keyword arguments for the dataset.
        kwargs_dataloader: Dictionary of keyword arguments for the DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
        """
    dataset = fed_dataset(train=train, **kwargs_dataset)
    return DataLoader(dataset, **kwargs_dataloader)


def get_element_from_dataloader(loader):
    """Extract all elements from a DataLoader into tensors.

    Args:
        loader: A PyTorch DataLoader instance.

    Returns:
        Tuple[Tensor, Tensor]: Concatenated input and label tensors.
    """
    if len(loader) == 0:
        return None, None
    X, Y = [], []
    for x, y in loader:
        X.append(x)
        Y.append(y)
    return torch.concat(X), torch.concat(Y)


def get_data_from_csv(dataset_name: str, batch_size) -> [List[torch.FloatTensor], List[torch.FloatTensor], bool]:
    """Load and preprocess data from CSV files for federated learning.

    Args:
        dataset_name (str): Name of the dataset to load.
        batch_size: Batch size for the DataLoaders.

    Returns:
        Tuple[List[DataLoader], List[DataLoader], List[DataLoader], bool]:
            - train_loaders: List of DataLoaders for training data.
            - val_loaders: List of DataLoaders for validation data.
            - test_loaders: List of DataLoaders for test data.
            - natural_split: Boolean indicating if the data split is natural.
    """
    root = get_path_to_datasets()
    if dataset_name == "liquid_asset":
        data_train, labels_train = prepare_liquid_asset(root, train=True)
    else:
        return ValueError("Dataset not recognized.")

    nb_of_clients = len(data_train)

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

    train_loaders = [DataLoader(TensorDataset(X_train[i], Y_train[i]), batch_size=batch_size) for i in
                     range(nb_of_clients)]
    val_loaders = [DataLoader(TensorDataset(X_val[i], Y_val[i]), batch_size=batch_size) for i in range(nb_of_clients)]
    test_loaders = [DataLoader(TensorDataset(X_test[i], Y_test[i]), batch_size=batch_size) for i in
                    range(nb_of_clients)]

    natural_split = True
    return train_loaders, val_loaders, test_loaders, natural_split

def generate_client_models(N: int, K: int, d: int, cluster_variance: float = 2):
    """Generate synthetic client models grouped into clusters.

    Args:
        N (int): Number of clients.
        K (int): Number of clusters.
        d (int): Dimensionality of the models.
        cluster_variance (float, optional): Variance within each cluster. Defaults to 2.

    Returns:
        Tuple[List[Tensor], List[Tensor]]: List of client models and their variations.
    """
    # Generate K cluster centers
    # Create an uninitialized tensor and fill it with values from the uniform distribution
    cluster_centers = [torch.empty(d).uniform_(-5, 5) for _ in range(K)]
    # Assign each client to a cluster and generate their model
    variations = [cluster_variance * torch.randn(d) for i in range(N)]
    return [cluster_centers[i % K] + variations[i] for i in range(N)], variations

def get_synth_data(batch_size: int, nb_clients = 4, nb_clusters = 1, dim: int = 2) -> tuple[
    list[DataLoader[Any]], list[DataLoader[Any]], list[DataLoader[Any]], bool]:
    """Generate synthetic data (LSR) for federated learning experiments.

    Args:
        batch_size (int): Batch size for the DataLoaders.
        nb_clients (int, optional): Number of clients. Defaults to 4.
        nb_clusters (int, optional): Number of clusters. Defaults to 1.
        dim (int, optional): Dimensionality of the data. Defaults to 2.

    Returns:
        Tuple[List[DataLoader], List[DataLoader], List[DataLoader], bool]:
            - train_loaders: List of DataLoaders for training data.
            - val_loaders: List of DataLoaders for validation data.
            - test_loaders: List of DataLoaders for test data.
            - natural_split: Boolean indicating if the data split is natural.
        """
    true_models, variations = generate_client_models(nb_clients, nb_clusters, dim, cluster_variance=0.1)
    datasets = [SyntheticLSRDataset(m, v, batch_size) for (m, v) in zip(true_models, variations)]

    train_loaders = [DataLoader(d, batch_size=None) for d in datasets]
    val_loaders = [DataLoader(d, batch_size=None) for d in datasets]
    test_loaders = [DataLoader(d, batch_size=None) for d in datasets]

    natural_split = True
    return train_loaders, val_loaders, test_loaders, natural_split


def get_data_from_pytorch(dataset_name: str, fed_dataset, nb_of_clients, split_type, kwargs_train_dataset, kwargs_test_dataset,
                          kwargs_dataloader) -> tuple[list[DataLoader[Any]], list[DataLoader[Any]], list[DataLoader[Any]], bool]:
    """Load and preprocess data from PyTorch datasets for federated learning.

    Args:
        dataset_name (str): Name of the dataset.
        fed_dataset: Federated dataset class.
        nb_of_clients: Number of clients.
        split_type: Type of data split (e.g., 'non-iid').
        kwargs_train_dataset: Keyword arguments for the training dataset.
        kwargs_test_dataset: Keyword arguments for the test dataset.
        kwargs_dataloader: Keyword arguments for the DataLoader.

    Returns:
        Tuple[List[DataLoader], List[DataLoader], List[DataLoader], bool]:
            - train_loaders: List of DataLoaders for training data.
            - val_loaders: List of DataLoaders for validation data.
            - test_loaders: List of DataLoaders for test data.
            - natural_split: Boolean indicating if the data split is natural.
    """
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
    X, Y = create_non_iid_split(X, Y, nb_of_clients, split_type=split_type, dataset_name=dataset_name)

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

    print_mem_usage()

    print("Removing the concatenated datasets.")
    del X, Y
    torch.cuda.empty_cache()
    gc.collect()

    print_mem_usage()

    train_loaders = [DataLoader(TensorDataset(X_train[i], Y_train[i]), **kwargs_dataloader) for i in range(nb_of_clients)]
    val_loaders = [DataLoader(TensorDataset(X_val[i], Y_val[i]), **kwargs_dataloader) for i in range(nb_of_clients)]
    test_loaders = [DataLoader(TensorDataset(X_test[i], Y_test[i]), **kwargs_dataloader) for i in range(nb_of_clients)]

    natural_split = False
    return train_loaders, val_loaders, test_loaders, natural_split


def get_data_from_flamby(fed_dataset, nb_of_clients, dataset_name: str, batch_size, kwargs_dataloader, debug: bool = False) \
        -> tuple[list[DataLoader[Any]], list[DataLoader[Any]], list[DataLoader[Any]], bool]:
    """Load and preprocess data from FLamby datasets for federated learning.

    Args:
        fed_dataset: Federated dataset class.
        nb_of_clients: Number of clients.
        dataset_name (str): Name of the dataset.
        batch_size: Batch size for the DataLoaders.
        kwargs_dataloader: Keyword arguments for the DataLoader.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.

    Returns:
        Tuple[List[DataLoader], List[DataLoader], List[DataLoader], bool]:
            - train_loaders: List of DataLoaders for training data.
            - val_loaders: List of DataLoaders for validation data.
            - test_loaders: List of DataLoaders for test data.
            - natural_split: Boolean indicating if the data split is natural.
        """

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

        # I don't use ValSet, should be removed.
        X_val.append(torch.concat([data_test]))
        Y_val.append(torch.concat([labels_test]))

        X_train.append(torch.concat([data_train]))
        Y_train.append(torch.concat([labels_train]))
        X_test.append(torch.concat([data_test]))
        Y_test.append(torch.concat([labels_test]))

    train_loaders = [DataLoader(TensorDataset(X_train[i], Y_train[i]), batch_size=batch_size) for i in
                     range(nb_of_clients)]
    val_loaders = [DataLoader(TensorDataset(X_val[i], Y_val[i]), batch_size=batch_size) for i in range(nb_of_clients)]
    test_loaders = [DataLoader(TensorDataset(X_test[i], Y_test[i]), batch_size=batch_size) for i in
                    range(nb_of_clients)]

    natural_split = True
    return train_loaders, val_loaders, test_loaders, natural_split


def preprocess(example, tokenizer):
    """Preprocess a single example for multiple-choice NLP tasks.

    Args:
        example: A dictionary containing the question and answer choices.
        tokenizer: A tokenizer instance from HuggingFace Transformers.

    Returns:
        dict: Tokenized inputs with labels for the multiple-choice task.
    """

    options = 'ABCD'
    indices = list(range(5))  # Indexing 0, 1, 2, 3, 4 for each option

    option_to_index = {option: index for option, index in
                       zip(options, indices)}  # Converting options to indices '''A to 0'''
    index_to_option = {index: option for option, index in
                       zip(options, indices)}  # Converting indices to options '''0 to A'''

    first_sentence = [example['Question']] * 4  # Repeating the same question 5 times
    second_sentence = []  # Creating list of possible answers
    for option in options:
        second_sentence.append(example[option])

    # The tokenizer converts the question and answers into 'tokens'.
    # 'tokens' are simply a sequence of integers in which each specific integer corresponds to
    # a word or subword that BERT is capable of comprehending
    tokenized_example = tokenizer(first_sentence, second_sentence, truncation=True)

    # Indexing label - A,B,C,D or E - to either 0, 1, 2, 3, 4, or 5
    tokenized_example['label'] = option_to_index[example['Answer']]

    # tokenized_example returns:
    # input_ids --> List of lists represents the tokens
    # token_type_ids --> A list of lists indicating whether each token belongs to the 1st or 2nd sentence
    # attention_mask --> List of list where each inner list is either 1 or 0. 1 are not padding tokens, 0 are padding tokens
    # label --> The index for the correct option
    return tokenized_example

def process_secqa(ds):
    """Process the SECQA dataset by removing unnecessary columns.

    Args:
        ds: The dataset to process.

    Returns:
        Dataset: The processed dataset.
    """
    return ds.remove_columns(['Explanation'])

def process_allenai(ds):
    """Process the AllenAI OpenBookQA dataset to match the expected format.

    Args:
        ds: The dataset to process.

    Returns:
        Dataset: The processed dataset.
    """
    def modify_allenai(example):
        choices_text = example["choices"]["text"]
        example["A"] = choices_text[0]
        example["B"] = choices_text[1]
        example["C"] = choices_text[2]
        example["D"] = choices_text[3]
        return example

    new_ds = ds.map(modify_allenai, remove_columns=["choices", "id"])
    new_ds = new_ds.rename_column("answerKey", "Answer")
    new_ds = new_ds.rename_column("question_stem", "Question")
    return new_ds

def process_cais(ds):
    """Process the CAIS MMLU dataset to match the expected format.

    Args:
        ds: The dataset to process.

    Returns:
        Dataset: The processed dataset.
    """
    options = 'ABCD'
    indices = list(range(5))  # Indexing 0, 1, 2, 3, 4 for each option

    option_to_index = {option: index for option, index in
                       zip(options, indices)}  # Converting options to indices '''A to 0'''
    index_to_option = {index: option for option, index in
                       zip(options, indices)}  # Converting indices to options '''0 to A'''

    def modify_cais(example):
        choices_text = example["choices"]
        example["A"] = choices_text[0]
        example["B"] = choices_text[1]
        example["C"] = choices_text[2]
        example["D"] = choices_text[3]
        example["Answer"] = index_to_option[example["answer"]]
        return example

    new_ds = ds.map(modify_cais, remove_columns=["choices", "subject", "answer"])
    new_ds = new_ds.rename_column("question", "Question")
    return new_ds

def get_data_from_NLP(batch_size: int) -> tuple[list[DataLoader[Any]], list[DataLoader[Any]], list[DataLoader[Any]], bool]:
    """Load and preprocess NLP datasets for multiple-choice tasks.

    Args:
        batch_size (int): Batch size for the DataLoaders.

    Returns:
        Tuple[List[DataLoader], List[DataLoader], List[DataLoader], bool]:
            - train_loaders: List of DataLoaders for training data.
            - val_loaders: List of DataLoaders for validation data.
            - test_loaders: List of DataLoaders for test data.
            - natural_split: Boolean indicating if the data split is natural.
    """

    paths = ["cais/mmlu", "cais/mmlu", "cais/mmlu", "cais/mmlu", "cais/mmlu", "cais/mmlu", "cais/mmlu"]
    names = ["abstract_algebra", "machine_learning", "computer_security", "anatomy", "philosophy", "sociology", "marketing"]
    dict_download = {"zefang-liu/secqa": ["test", "val", "dev"], "allenai/openbookqa": ["train", "validation", "test"],
            "cais/mmlu": ["test", "validation", "dev"]}

    dict_processing = {"zefang-liu/secqa": process_secqa, "allenai/openbookqa": process_allenai,
                       "cais/mmlu": process_cais}

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    train_loaders, val_loaders, test_loaders = [], [], []

    for path, name in zip(paths, names):
        print(path)
        ds = load_dataset(path, name)
        ds = dict_processing[path](ds)

        train_ds = ds[dict_download[path][0]]
        val_ds = ds[dict_download[path][1]]
        test_ds = ds[dict_download[path][2]]

        # Dataset.map() method in the HuggingFace datasets library does not support passing additional keyword arguments
        # directly to the function being mapped
        tokenized_train_ds = train_ds.map(lambda x: preprocess(x, tokenizer), batched=False,
                                          remove_columns=['Question', 'A', 'B', 'C', 'D', 'Answer'])
        tokenized_val_ds = val_ds.map(lambda x: preprocess(x, tokenizer), batched=False,
                                      remove_columns=['Question', 'A', 'B', 'C', 'D', 'Answer'])
        tokenized_test_ds = test_ds.map(lambda x: preprocess(x, tokenizer), batched=False,
                                      remove_columns=['Question', 'A', 'B', 'C', 'D', 'Answer'])

        tokenized_train_ds.set_format("torch")
        tokenized_val_ds.set_format("torch")
        tokenized_test_ds.set_format("torch")

        train_loader = DataLoader(
            tokenized_train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        val_loader = DataLoader(
            tokenized_val_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        test_loader = DataLoader(
            tokenized_test_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)

    natural_split = True
    return train_loaders, val_loaders, test_loaders, natural_split