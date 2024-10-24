from typing import Callable
import torch
import numpy as np
from shapely.geometry import LineString
from shapely import frechet_distance as shapely_frechet
from shapely import hausdorff_distance as shapely_hausdorff


def trajectory_metric(
    distance_function: Callable[[LineString, LineString], float | np.ndarray],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    reduction: str = "none"
) -> torch.Tensor:
    """
    Compute the trajectory-wise distance between two sets of trajectories using a given distance function.

    Args:
        distance_function: A function that takes two LineString objects and returns the distance between them.
        y_true: A tensor of shape (batch_size, seq_len, 2) representing the true trajectories.
        y_pred: A tensor of shape (batch_size, seq_len, 2) representing the predicted trajectories.
        reduction: Specifies the reduction to apply to the output. One of {"none", "mean", "sum"}.

    Returns:
        - Tensor of shape (batch_size,) if reduction is "none".
        - Tensor of shape (1,) if reduction is "mean" or "sum".
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    assert y_true.shape[-1] == 2, "y_true and y_pred must have shape (batch_size, seq_len, 2)"

    distances = torch.zeros(y_true.shape[0], dtype=y_true.dtype)
    device = y_true.device
    for i in range(y_true.shape[0]):
        y_true_string = LineString(y_true[i].cpu().detach().numpy())
        y_pred_string = LineString(y_pred[i].cpu().detach().numpy())
        distances[i] = distance_function(y_true_string, y_pred_string)

    distances = distances.to(device)

    if reduction == "mean":
        return distances.mean()
    if reduction == "sum":
        return distances.sum()
    return distances


def frechet_distance(y_true: torch.Tensor, y_pred: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    """
    Compute the trajectory-wise Frechet distance between two sets of trajectories.

    Args:
        y_true: A tensor of shape (batch_size, seq_len, 2) representing the true trajectories.
        y_pred: A tensor of shape (batch_size, seq_len, 2) representing the predicted trajectories.
        reduction: Specifies the reduction to apply to the output. One of {"none", "mean", "sum"}.

    Returns:
        - Tensor of shape (batch_size,) if reduction is "none".
        - Tensor of shape (1,) if reduction is "mean" or "sum".
    """
    return trajectory_metric(shapely_frechet, y_true, y_pred, reduction=reduction)
    

def hausdorff_distance(y_true: torch.Tensor, y_pred: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    """
    Compute the trajectory-wise Hausdorff distance between two sets of trajectories.

    Args:
        y_true: A tensor of shape (batch_size, seq_len, 2) representing the true trajectories.
        y_pred: A tensor of shape (batch_size, seq_len, 2) representing the predicted trajectories.
        reduction: Specifies the reduction to apply to the output. One of {"none", "mean", "sum"}.

    Returns:
        - Tensor of shape (batch_size,) if reduction is "none".
        - Tensor of shape (1,) if reduction is "mean" or "sum".
    """
    return trajectory_metric(shapely_hausdorff, y_true, y_pred, reduction=reduction)
