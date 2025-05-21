from typing import List

import numpy as np

from src.utils.PickleHandler import pickle_saver
from src.utils.Utilities import create_folder_if_not_existing


class Metrics:

    def __init__(self, dataset_name: str, distance_name: str, nb_of_clients: int,
                 nb_of_points_by_clients: List[int]) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.distance_name = distance_name
        self.nb_of_clients = nb_of_clients
        self.nb_of_points_by_clients = nb_of_points_by_clients
        self.distances_heter = []

    def reinitialize(self):
        self.distances_heter = []

    def add_distances(self, distances_heter):
        self.distances_heter.append(distances_heter)

        create_folder_if_not_existing("pickle/{0}/matrix_of_distances".format(self.dataset_name))
        pickle_saver(self, "pickle/{0}/matrix_of_distances/{1}".format(self.dataset_name, self.distance_name))

    def aggreage_heter(self):
        return np.mean(self.distances_heter, axis=0)