"""Created by Constantin Philippenko, 30th July 2024."""

import lifelines
import numpy as np


def accuracy(y_true, y_pred):
    pass


def auc(y_true, y_pred):
    # y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        # return roc_auc_score(y_true, y_pred)
        # proposed modification in order to get a metric that calcs on center 2
        # (y=1 only on that center)
        return ((y_pred > 0.5) == y_true).mean()
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