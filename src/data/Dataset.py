"""
Created by Philippenko.

This class contains the data processing for datasets used in a deeplearning environment that are not classical
(e.g. QRT Data Challenge).
"""

import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def prepare_liquid_asset(root: str, train = True):

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    if not train:
        return None, None

    X_raw_train = pd.read_csv(f'{root}/liquid_asset/X_train.csv')
    Y_raw_train = pd.read_csv(f'{root}/liquid_asset/y_train.csv', index_col=0)

    # Taking the asset name of features and predictions.
    asset_feature = X_raw_train.columns.values[2:-1]
    asset_predict = ['RET_' + str(id) for id in X_raw_train['ID_TARGET'].unique()]
    assert len(asset_feature) == len(
        asset_predict), "The lenghts are not equal while they should both be equal to 100."

    Y_value = {}

    for day in X_raw_train.ID_DAY.unique():
        b = Y_raw_train[X_raw_train.ID_DAY == day]['RET_TARGET']
        b.index = ['RET_' + str(int(j)) for j in X_raw_train.loc[X_raw_train.ID_DAY == day].ID_TARGET]
        Y_value[day] = b
    Y_value = pd.DataFrame(Y_value).T.astype(float)

    X = X_raw_train.drop_duplicates(subset=asset_feature).drop(columns=["ID", "ID_TARGET"]).set_index("ID_DAY")
    assert len(X) == len(Y_value), "Features and multilabel output are not identical."

    print(f"There is {X.isna().sum().sum()} nan value in the X dataset.")
    # The nan value in Y corresponds to days for which we have no measure for a given asset.
    print(f"There is {Y_value.isna().sum().sum()} nan value in the Y dataset.")

    # We replace the NaN in the features by the mean.
    for asset in asset_feature:
        X[asset] = X[asset].fillna(X[asset].mean())
    print(f"There is now {X.isna().sum().sum()} nan value in the X dataset.")

    numerical_transformer.fit(X)
    X = numerical_transformer.transform(X)

    X_clients, Y_clients = [], []
    for asset in asset_predict:
        n = len(Y_value[Y_value[asset].notna()][asset])
        X_clients.append(torch.tensor(X[Y_value[asset].notna()]).float())
        Y_clients.append(torch.tensor(Y_value[Y_value[asset].notna()][asset].values).float().reshape(n, 1))

    return X_clients, Y_clients