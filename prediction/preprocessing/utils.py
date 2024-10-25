from datetime import timedelta
from typing import cast

import numpy as np
import torch


def timedelta_to_seconds(delta: timedelta):
    """This function converts a timedelta object to seconds.

    Args:
        delta: Timedelta object to convert to seconds
    """
    total_seconds = delta.total_seconds()
    total_microseconds = delta.microseconds
    return total_seconds + (total_microseconds / 1000000)


def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """This function calculates the haversine distance between two points given their coordinates.

    Args:
        lat1: Latitude of the first point
        lng1: Longitude of the first point
        lat2: Latitude of the second point
        lng2: Longitude of the second point

    Returns:
        The haversine distance between the two points in kilometers.
    """
    lat1, lng1, lat2, lng2 = map(np.deg2rad, [lat1, lng1, lat2, lng2])
    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1
    a = (np.sin(delta_lat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(delta_lng / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * cast(float, c)  # haversine distance


def haversine_tensor(
    traj1: torch.Tensor, traj2: torch.Tensor, traj_reduction: str = "sum", tensor_reduction: str = "mean"
) -> torch.Tensor:
    """Calculate the pairwise haversine distance between two sets of trajectories.

    Args:
        traj1: Tensor of shape (batch_size, seq_len, 2) containing the first trajectories
        traj2: Tensor of shape (batch_size, seq_len, 2) containing the second trajectories
        traj_reduction: Reduction method for trajectory points. Can be "none", "sum" or "mean".
        tensor_reduction: Reduction method for batch dimension. Can be "none", "sum" or "mean".

    Returns:
        Tensor with shape depending on reduction methods:
        - (batch_size, seq_len) if traj_reduction="none" and tensor_reduction="none"
        - (seq_len,) if traj_reduction="none" and tensor_reduction="sum"/"mean"
        - (batch_size,) if traj_reduction="sum"/"mean" and tensor_reduction="none"
        - scalar if traj_reduction="sum"/"mean" and tensor_reduction="sum"/"mean"
    """
    assert traj1.shape == traj2.shape

    lat1, lng1 = traj1[..., 0], traj1[..., 1]
    lat2, lng2 = traj2[..., 0], traj2[..., 1]

    lat1, lng1, lat2, lng2 = map(torch.deg2rad, [lat1, lng1, lat2, lng2])

    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1

    a = (torch.sin(delta_lat / 2) ** 2) + torch.cos(lat1) * torch.cos(lat2) * (torch.sin(delta_lng / 2) ** 2)
    c = 2 * torch.arctan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = 6371 * c  # haversine distance in kilometers

    if traj_reduction == "sum":
        distance = distance.sum(dim=1)
    elif traj_reduction == "mean":
        distance = distance.mean(dim=1)

    if tensor_reduction == "sum":
        distance = distance.sum(dim=0)
    elif tensor_reduction == "mean":
        distance = distance.mean(dim=0)

    return distance


def calc_sog(lat1: float, lng1: float, lat2: float, lng2: float, time: float) -> float:
    """This function calculates the speed between two points given their coordinates and time difference.

    The calculation is based on the haversine distance formula between two points on the Earth's surface
    divided by the time difference between the two points.

    Args:
        lat1: Latitude of the first point
        lng1: Longitude of the first point
        lat2: Latitude of the second point
        lng2: Longitude of the second point
        time: Time difference between the two points in seconds
    """
    distance = haversine(lat1, lng1, lat2, lng2)
    return distance / (time + 1e-10)  # add small number to avoid division by zero
