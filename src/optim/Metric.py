"""Created by Constantin Philippenko, 30th July 2024."""

import lifelines
import numpy as np
import torch

import torch.nn.functional as F


def accuracy(y_true, y_output_net):
    """
    Calculate the accuracy of predictions.

    Parameters:
    y_true (list or array-like): True labels
    y_output_net (list or array-like): Output of the neural network (without softmax function).

    Returns:
    float: Accuracy as the proportion of correct predictions
    """

    # Apply softmax to get probabilities
    test_probabilities = F.softmax(y_output_net, dim=1)

    # Get the predicted class (the class with the highest probability)
    predicted_label = torch.argmax(test_probabilities, dim=1)

    # Ensure both inputs are lists or arrays of the same length
    assert len(y_true) == len(predicted_label), "Length of y_true and y_output_net must be the same"

    # Calculate the number of correct predictions
    correct = sum(yt == yp for yt, yp in zip(y_true, predicted_label))

    # Calculate accuracy as the proportion of correct predictions
    accuracy_value = correct / len(y_true)

    return accuracy_value


def auc(y_true, y_pred):
    # y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be the same"
    try:
        # return roc_auc_score(y_true, y_pred)
        # proposed modification in order to get a metric that calcs on center 2
        # (y=1 only on that center)
        return ((y_pred > 0.5) == y_true).sum()/len(y_pred)
    except ValueError:
        return np.nan


def c_index(y_true, y_pred):
    """Calculates the concordance index (c-index) between a series of event
    times and a predicted score.
    The c-index is the average of how often a model says X is greater than Y
    when, in the observed data, X is indeed greater than Y.
    The c-index also handles how to handle censored values.
    Parameters
    ----------
    y_true : numpy array of floats of dimension (n_samples, 2), real
            survival times from the observational data
    pred : numpy array of floats of dimension (n_samples, 1), predicted
            scores from a model
    Returns
    -------
    c-index: float, calculating using the lifelines library
    """

    c_index = lifelines.utils.concordance_index(y_true[:, 1], -y_pred, y_true[:, 0])

    return c_index

def dice(y_true, y_pred):
    """Soft Dice coefficient."""
    SPATIAL_DIMENSIONS = 2, 3, 4
    intersection = (y_pred * y_true).sum(axis=SPATIAL_DIMENSIONS)
    union = (0.5 * (y_pred + y_true)).sum(axis=SPATIAL_DIMENSIONS)
    dice = intersection / (union + 1.0e-7)
    # If both inputs are empty the dice coefficient should be equal 1
    dice[union == 0] = 1
    return np.mean(dice)