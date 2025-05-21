"""
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
        mean = X[asset].mean()
        X[asset] = X[asset].fillna(mean)
        # X_raw_test[asset] = X_raw_test[asset].fillna(mean)
    print(f"There is now {X.isna().sum().sum()} nan value in the X dataset.")

    numerical_transformer.fit(X)
    X = numerical_transformer.transform(X)
    # X_test = numerical_transformer.transform(X_raw_test)

    X_clients, Y_clients = [], []
    for asset in asset_predict:
        n = len(Y_value[Y_value[asset].notna()][asset])
        X_clients.append(torch.tensor(X[Y_value[asset].notna()]).float())
        Y_clients.append(torch.tensor(Y_value[Y_value[asset].notna()][asset].values).float().reshape(n, 1))

    return X_clients, Y_clients


def load_liquid_dataset_test(root: str):
    """
    Load and preprocess the liquid asset dataset.

    Args:
        root (str): The directory containing the dataset files.

    Returns:
        X (DataFrame): Preprocessed training data.
        X_raw_test (DataFrame): Preprocessed test data.
        numerical_transformer (Pipeline): Fitted numerical transformer.
    """
    # Define the numerical transformer
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Load the datasets
    X_raw_train = pd.read_csv(f'{root}/liquid_asset/X_train.csv')
    X_raw_test = pd.read_csv(f'{root}/liquid_asset/X_test.csv')

    # Get asset feature columns and prediction targets
    asset_feature = X_raw_train.columns.values[2:-1]
    asset_predict = ['RET_' + str(id) for id in X_raw_train['ID_TARGET'].unique()]

    # Validate dimensions
    assert len(asset_feature) == len(asset_predict), "Asset feature and target length mismatch."

    # Fill missing values in the training and test datasets
    for asset in asset_feature:
        mean = X_raw_train[asset].mean()
        X_raw_train[asset] = X_raw_train[asset].fillna(mean)
        X_raw_test[asset] = X_raw_test[asset].fillna(mean)

    # Fit the numerical transformer on the training data
    numerical_transformer.fit(X_raw_train.drop(columns=["ID", "ID_TARGET"]).set_index("ID_DAY"))

    # Return the processed training and test datasets, and the transformer
    return X_raw_train, X_raw_test, numerical_transformer


def do_prediction_liquid_asset(network, X_raw_train, dataset_for_prediction, numerical_transformer):
    """
    Perform predictions on the test dataset using the trained PyTorch models.

    Args:
        network: The PyTorch network with clients containing trained models.
        X_raw_train (DataFrame): The raw training dataset.
        dataset_for_prediction (DataFrame): The dataset to make predictions on.
        numerical_transformer (Pipeline): The transformer used for feature scaling.

    Returns:
        None. Saves predictions to 'benchmark.csv'.
    """
    # Initialize dictionary for predictions
    df_list = []
    id_targets = list(X_raw_train['ID_TARGET'].unique())

    # Make predictions for each row in the test dataset
    for asset_idx in id_targets:
        subset = dataset_for_prediction[dataset_for_prediction["ID_TARGET"] == asset_idx]
        # Drop unused columns from the test dataset
        X_pred = subset.drop(columns=["ID", "ID_DAY", "ID_TARGET"])
        # Apply the transformer to the test data
        X_pred_transformed = numerical_transformer.transform(X_pred)

        client_idx = id_targets.index(asset_idx)

        # Get the corresponding client model and device
        model = network.clients[client_idx].trained_model
        device = network.clients[client_idx].device

        # Set model to evaluation mode
        model.eval()

        # Make prediction with the model
        with torch.no_grad():
            predictions = torch.sign(model(torch.tensor(X_pred_transformed).float().to(device)))

        # Convert the predictions to a DataFrame, using 'ID' as the index
        predictions_df = pd.DataFrame(predictions.cpu().numpy(), index=subset['ID'], columns=[f"RET_TARGET"]).astype(int)

        # Append the predictions DataFrame to the list
        df_list.append(predictions_df)

    # Concatenate all individual DataFrames into a single DataFrame
    final_predictions = pd.concat(df_list).sort_index()
    final_predictions.index.name = None
    final_predictions.to_csv('./benchmark.csv')