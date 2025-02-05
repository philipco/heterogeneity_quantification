"""Created by Constantin Philippenko, 29th September 2022."""
from typing import List

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from src.data.DataCollatorForMultipleChoice import DataCollatorForMultipleChoice
from src.data.Dataset import prepare_liquid_asset
from src.data.DatasetConstants import CHECKPOINT
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


def get_data_from_csv(dataset_name: str, batch_size) -> [List[torch.FloatTensor], List[torch.FloatTensor], bool]:

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


def get_synth_data(dataset_name: str, batch_size) -> [List[torch.FloatTensor], List[torch.FloatTensor], bool]:

    n = 200
    X1, X2, Y1, Y2 = f("same_partionned_support", n=n)
    data_train = [torch.Tensor(X1).reshape(n,1), torch.Tensor(X2).reshape(n,1)]
    labels_train = [torch.Tensor(Y1).reshape(n, 1), torch.Tensor(Y2).reshape(n, 1)]

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


def get_data_from_pytorch(fed_dataset, nb_of_clients, split_type, batch_size, kwargs_train_dataset, kwargs_test_dataset,
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
    X, Y = create_non_iid_split(X, Y, nb_of_clients, split_type=split_type)

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

    train_loaders = [DataLoader(TensorDataset(X_train[i], Y_train[i]), batch_size=batch_size) for i in range(nb_of_clients)]
    val_loaders = [DataLoader(TensorDataset(X_val[i], Y_val[i]), batch_size=batch_size) for i in range(nb_of_clients)]
    test_loaders = [DataLoader(TensorDataset(X_test[i], Y_test[i]), batch_size=batch_size) for i in range(nb_of_clients)]

    natural_split = False
    return train_loaders, val_loaders, test_loaders, natural_split


def get_data_from_flamby(fed_dataset, nb_of_clients, dataset_name: str, batch_size, kwargs_dataloader, debug: bool = False) \
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
        # Therefore, the val and test set are IDENTICAL for TCGA_BRCA.
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

    train_loaders = [DataLoader(TensorDataset(X_train[i], Y_train[i]), batch_size=batch_size) for i in
                     range(nb_of_clients)]
    val_loaders = [DataLoader(TensorDataset(X_val[i], Y_val[i]), batch_size=batch_size) for i in range(nb_of_clients)]
    test_loaders = [DataLoader(TensorDataset(X_test[i], Y_test[i]), batch_size=batch_size) for i in
                    range(nb_of_clients)]

    natural_split = True
    return train_loaders, val_loaders, test_loaders, natural_split


def preprocess(example, tokenizer):

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
    return ds.remove_columns(['Explanation'])

def process_allenai(ds):
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

def get_data_from_NLP(batch_size: int):

    #paths = ["zefang-liu/secqa", "allenai/openbookqa", "cais/mmlu"]
    paths = ["cais/mmlu", "cais/mmlu", "cais/mmlu", "cais/mmlu", "cais/mmlu", "cais/mmlu", "cais/mmlu"]
    names = ["abstract_algebra", "machine_learning", "computer_security", "anatomy", "philosophy", "sociology", "marketing"]
    #names = ["secqa_v1", "main", "machine_learning"]
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