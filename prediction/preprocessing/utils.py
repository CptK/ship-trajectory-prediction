from datetime import timedelta
import numpy as np


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
    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1
    a = (np.sin(delta_lat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(delta_lng / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * c # haversine distance


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
    return distance / (time + 1e-10) # add small number to avoid division by zero