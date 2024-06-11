"""Created by Constantin Philippenko, 5th April 2022."""

import os
import pickle


def pickle_saver(data, filename: str) -> None:
    """Save a python object into a pickle file.
    If a file with the same name already exists, remove it.
    Store the file into a folder pickle/ which need to already exist.
    Args:
        data: the python object to save.
        filename: the filename where the object is saved.
    """
    file_to_save = "{0}.pkl".format(filename)
    if os.path.exists(file_to_save):
        os.remove(file_to_save)
    pickle_out = open(file_to_save, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def pickle_loader(filename: str):
    """Load a python object saved with pickle.
    Args:
        filename: the file where the object is stored.
    Returns:
        The python object to load.
    """
    pickle_in = open("{0}.pkl".format(filename), "rb")
    return pickle.load(pickle_in)