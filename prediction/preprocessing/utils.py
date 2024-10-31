from typing import cast

import numpy as np
import torch
from fastdtw.fastdtw import fastdtw


def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """This function calculates the haversine distance between two points given their coordinates.

    Args:
        lat1: Latitude of the first point within [-90, 90]
        lng1: Longitude of the first point within [-180, 180]
        lat2: Latitude of the second point within [-90, 90]
        lng2: Longitude of the second point within [-180, 180]

    Returns:
        The haversine distance between the two points in kilometers.
    """
    if not (-90 <= lat1 <= 90):
        print(f"lat1: {lat1}, lng1: {lng1}, lat2: {lat2}, lng2: {lng2}")
        raise ValueError("Latitude of the first point must be within [-90, 90]")
    if not (-90 <= lat2 <= 90):
        print(f"lat1: {lat1}, lng1: {lng1}, lat2: {lat2}, lng2: {lng2}")
        raise ValueError("Latitude of the second point must be within [-90, 90]")
    if not (-180 <= lng1 <= 180):
        print(f"lat1: {lat1}, lng1: {lng1}, lat2: {lat2}, lng2: {lng2}")
        raise ValueError("Longitude of the first point must be within [-180, 180]")
    if not (-180 <= lng2 <= 180):
        print(f"lat1: {lat1}, lng1: {lng1}, lat2: {lat2}, lng2: {lng2}")
        raise ValueError("Longitude of the second point must be within [-180, 180]")

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


def dtw_spatial(traj1: np.ndarray, traj2: np.ndarray) -> tuple[float, list]:
    """
    Calculate DTW distance between trajectories based on spatial coordinates.

    Args:
        traj1: First trajectory array with shape (n_points, 4) containing:
              - lat: Latitude in decimal degrees
              - lon: Longitude in decimal degrees
              - sog: Speed over ground in knots
              - cog: Course over ground in degrees [0, 360)
        traj2: Second trajectory array with shape (m_points, 4), same format

    Returns:
        DTW distance based on Haversine distances between points

    Example:
        >>> traj1 = np.array([[48.5, -125.5, 12.5, 180.0],
        ...                   [48.6, -125.4, 12.3, 182.0]])
        >>> traj2 = np.array([[48.4, -125.6, 11.5, 178.0],
        ...                   [48.5, -125.5, 11.8, 179.0]])
        >>> distance = dtw_spatial(traj1, traj2)
    """
    return cast(
        tuple[float, list],
        fastdtw(traj1[:, :2], traj2[:, :2], dist=lambda x, y: haversine(x[0], x[1], y[0], y[1])),
    )


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

    Returns:
        The speed between the two points in km/h.
    """
    if time < 0:
        raise ValueError("Time difference must be non-negative")
    if not (-90 <= lat1 <= 90):
        raise ValueError("Latitude of the first point must be within [-90, 90]")
    if not (-90 <= lat2 <= 90):
        raise ValueError("Latitude of the second point must be within [-90, 90]")
    if not (-180 <= lng1 <= 180):
        raise ValueError("Longitude of the first point must be within [-180, 180]")
    if not (-180 <= lng2 <= 180):
        raise ValueError("Longitude of the second point must be within [-180, 180]")

    if (lat1 == lat2 and lng1 == lng2) or time == 0:
        return 0.0

    distance = haversine(lat1, lng1, lat2, lng2)
    return distance / (time + 1e-10) * 3600  # convert to km/h, add small number to avoid division by zero


def azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate azimuth angle between two points on Earth.
    
    The azimuth is the angle between the north vector and the direction to 
    the target point, measured clockwise in degrees.
    
    Args:
        lat1: Starting point latitude in decimal degrees
        lon1: Starting point longitude in decimal degrees
        lat2: Ending point latitude in decimal degrees
        lon2: Ending point longitude in decimal degrees
        
    Returns:
        Azimuth angle in degrees [0, 360)
        
    Example:
        >>> azimuth = calculate_azimuth(48.0, -125.0, 48.1, -125.1)
        >>> print(f"Azimuth angle: {azimuth:.1f} degrees")
    """
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    
    dlon = lon2 - lon1
    
    x = np.sin(dlon) * np.cos(lat2)
    y = (np.cos(lat1) * np.sin(lat2) - 
         np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    
    azimuth = np.degrees(np.arctan2(x, y))
    return (azimuth + 360) % 360
