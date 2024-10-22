import pandas as pd
from shapely.geometry import LineString, Point
from tqdm import tqdm


def _create_geometry(group, min_points: int) -> LineString | None:
    """Create a LineString from the given group of points.

    If the group has less than min_points points, return None.

    Args:
        group: The group of points to create the LineString from.
        min_points: The minimum number of points required to create the LineString.

    Returns:
        The LineString created from the group of points, or None if the group has less than min_points points
    """
    points = [Point(row["LON"], row["LAT"]) for _, row in group.iterrows()]
    if len(points) >= min_points:
        return LineString(points)
    else:
        return None


def build_trajectories(
    df: pd.DataFrame,
    min_points: int,
    vessel_groups: dict[int, str],
    verbose: bool = True
) -> pd.DataFrame:
    """Build trajectories from the given DataFrame.

    Args:
        df: The DataFrame to build the trajectories from.
        min_points: The minimum number of points required to create a trajectory.
        vessel_groups: A dictionary mapping vessel types to their names.
        verbose: Whether to display a progress bar.

    Returns:
        A DataFrame containing the built trajectories.
    """
    if verbose:
        tqdm.pandas()

    func = df.groupby('MMSI').progress_apply if verbose else df.groupby('MMSI').apply

    result = func(
        lambda x: pd.Series({
            'geometry': _create_geometry(x, min_points),
            'mmsi': x["MMSI"].iloc[0],
            'velocities': x["SOG"].tolist() if len(x) >= min_points else None,
            'orientations': x["COG"].tolist() if len(x) >= min_points else None,
            'start_time': x["BaseDateTime"].min() if len(x) >= min_points else None,
            'end_time': x["BaseDateTime"].max() if len(x) >= min_points else None,
            'point_count': len(x),
            'vessel_type': vessel_groups[int(x["VesselType"].iloc[0])],
            'timestamps': x["BaseDateTime"].tolist() if len(x) >= min_points else None
        }) if len(x) >= min_points else None
    ).reset_index()

    return result.dropna()
