"""Created by Constantin Philippenko, 29th September 2022."""
from typing import List

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from src.data.DatasetConstants import PCA_NB_COMPONENTS
from src.data.Data import DataCentralized
from src.data.Split import iid_split
from src.utils.PickleHandler import pickle_saver
from src.utils.Utilities import create_folder_if_not_existing, get_project_root
from src.utils.UtilitiesNumpy import fit_PCA, compute_PCA



def centralized_processing_of_data(dataset_name: str, features: List[np.array], labels: List[np.array],
                                   batch_size: int, nb_of_clients: int) -> DataCentralized:

    # Scaling and computing a PCA for the complete dataset once for all.
    scaler = StandardScaler().fit(np.concatenate(features))
    ipca_data = IncrementalPCA(n_components=PCA_NB_COMPONENTS)
    ipca_data = fit_PCA(np.concatenate(features), ipca_data, scaler, batch_size)
    features = [compute_PCA(f, ipca_data, scaler, batch_size) for f in features]

    nb_points_by_clients = [len(l) for l in labels]
    features_iid, labels_iid = iid_split(np.concatenate(features), np.concatenate(labels), nb_of_clients,
                                         nb_points_by_clients)

    data = DataCentralized(dataset_name, nb_points_by_clients, features_iid, features, labels_iid, labels)

    root = get_project_root()
    create_folder_if_not_existing("{0}/pickle/{1}/processed_data".format(root, dataset_name))
    pickle_saver(data, "{0}/pickle/{1}/processed_data/centralized".format(root, dataset_name))

    return data