"""
Metrics for evaluating predictions in various supervised learning settings.

This module includes functions for:
- Classification accuracy (with softmax).
- Approximate AUC (classification).
- Concordance index (c-index) for survival analysis.
- Dice coefficient (typically for segmentation tasks).
- L1-weighted directional accuracy.
- Normalized MSE.
"""

import lifelines
import numpy as np
import torch
import torch.nn.functional as F


def accuracy(y_true, y_output_net):
    """
    Compute classification accuracy based on raw neural network outputs.

    Applies softmax to the raw logits, computes predicted labels,
    and compares them to ground truth.

    Parameters:
        y_true (list or array-like): True class labels.
        y_output_net (Tensor): Raw outputs from the network, shape (N, num_classes).

    Returns:
        float: Accuracy (proportion of correct predictions).
    """

    # Apply softmax to logits to get class probabilities
    test_probabilities = F.softmax(y_output_net, dim=1)

    # Get the index of the maximum probability (predicted class)
    predicted_label = torch.argmax(test_probabilities, dim=1)

    assert len(y_true) == len(predicted_label), \
        "Length of y_true and y_output_net must be the same"

    # Count number of correct predictions
    correct = sum(yt == yp for yt, yp in zip(y_true, predicted_label))

    # Compute proportion of correct predictions
    accuracy_value = correct / len(y_true)
    return accuracy_value


def auc(y_true, y_pred):
    """
    Approximate Area Under the Curve (AUC) using a fixed threshold (0.5).

    This is not the ROC-AUC, but a simplified metric often useful when
    one class is significantly more represented (e.g., in center 2).

    Parameters:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted probabilities or scores.

    Returns:
        float: Approximate AUC metric (accuracy under thresholding).
    """

    assert len(y_true) == len(y_pred), \
        "Length of y_true and y_pred must be the same"

    try:
        return ((y_pred > 0.5) == y_true).sum() / len(y_pred)
    except ValueError:
        # Can occur if all predictions belong to one class (in small batches)
        return np.nan


def c_index(y_true, y_pred):
    """
    Compute the concordance index (c-index) for survival models.

    This metric evaluates the agreement between predicted risk scores
    and observed survival durations, taking into account censoring.

    Parameters:
        y_true (Tensor): Tensor of shape (n_samples, 2),
                         where [:, 0] is event indicator (1 if observed),
                         and [:, 1] is survival/censoring time.
        y_pred (Tensor): Predicted risk scores (larger = more at risk).

    Returns:
        float: Concordance index, computed via `lifelines.utils.concordance_index`.
    """

    # Convert tensors to NumPy arrays for lifelines compatibility
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    # Note: We negate predictions as lifelines assumes higher risk = lower survival time
    return lifelines.utils.concordance_index(
        y_true[:, 1], -y_pred, y_true[:, 0]
    )


def dice(y_true, y_pred):
    """
    Compute the soft Dice coefficient between predicted and ground truth tensors.

    The Dice coefficient is often used in segmentation tasks to evaluate spatial overlap.

    Parameters:
        y_true (Tensor): Ground truth segmentation mask.
        y_pred (Tensor): Predicted mask (can be probabilistic or binary).

    Returns:
        Tensor: Mean Dice coefficient across batch.
    """

    # Define the spatial dimensions (batch, channel, H, W, D)
    SPATIAL_DIMENSIONS = 2, 3, 4

    # Compute intersection and soft union
    intersection = (y_pred * y_true).sum(dim=SPATIAL_DIMENSIONS)
    union = 0.5 * (y_pred + y_true).sum(dim=SPATIAL_DIMENSIONS)

    # Compute soft Dice per sample
    dice = intersection / (union + 1.0e-7)

    # Special case: if both inputs are empty, return dice = 1
    dice[union == 0] = 1
    return torch.mean(dice)


def l1_accuracy(y_true, y_pred):
    """
    Directional accuracy weighted by L1 norm of the target.

    The score emphasizes correctness of sign and penalizes errors
    in high-magnitude targets.

    Parameters:
        y_true (Tensor): Ground truth tensor.
        y_pred (Tensor): Predicted values.

    Returns:
        Tensor: L1-weighted accuracy.
    """

    # Compute correctness of signs and weight by absolute target values
    metric = torch.abs(y_true) * (torch.sign(y_true) == torch.sign(y_pred))

    # Normalize by total L1 norm of target
    return torch.sum(metric) / torch.norm(y_true, p=1)


def mse_accuracy(y_true, y_pred):
    """
    Mean squared error divided by number of samples.

    Parameters:
        y_true (Tensor): Ground truth tensor.
        y_pred (Tensor): Predicted values.

    Returns:
        Tensor: Average squared L2 error per sample.
    """
    return torch.mean(torch.norm(y_true - y_pred, p=2)**2) / len(y_true)
