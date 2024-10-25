from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

import pandas as pd
from shapely.geometry import LineString, Point
from tqdm import tqdm

from prediction.data.loader import load_data


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
    verbose: bool = True,
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

    df["VesselType"] = df["VesselType"].apply(lambda x: x if x in vessel_groups.keys() else 21)
    func = df.groupby("MMSI").progress_apply if verbose else df.groupby("MMSI").apply

    result = func(
        lambda x: (
            pd.Series(
                {
                    "geometry": _create_geometry(x, min_points),
                    "mmsi": x["MMSI"].iloc[0],
                    "velocities": x["SOG"].tolist() if len(x) >= min_points else None,
                    "orientations": x["COG"].tolist() if len(x) >= min_points else None,
                    "start_time": x["BaseDateTime"].min() if len(x) >= min_points else None,
                    "end_time": x["BaseDateTime"].max() if len(x) >= min_points else None,
                    "point_count": len(x),
                    "vessel_type": vessel_groups[int(x["VesselType"].iloc[0])],
                    "timestamps": x["BaseDateTime"].tolist() if len(x) >= min_points else None,
                }
            )
            if len(x) >= min_points
            else None
        )
    ).reset_index()

    return result.dropna()


def _load_and_build_single(
    date: datetime, min_points: int, vessel_groups: dict[int, str], verbose: bool = True
) -> pd.DataFrame:
    """Load the data for the given date and build the trajectories.

    Args:
        date: The date to load the data for.
        min_points: The minimum number of points required to create a trajectory.
        vessel_groups: A dictionary mapping vessel types to their names.
        verbose: Whether to display a progress bar.

    Returns:
        A DataFrame containing the built trajectories.
    """
    print(f"Loading and building trajectories for {date}")
    df = load_data(date)
    return build_trajectories(df, min_points, vessel_groups, verbose)


def load_and_build(
    start_date: datetime,
    end_date: datetime,
    min_points: int,
    vessel_groups: dict[int, str],
    n_processes: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    n_processes = n_processes if n_processes is not None else max(cpu_count() - 1, 1)
    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    if verbose:
        print(f"Loading and building trajectories for {len(all_dates)} days using {n_processes} processes")

    with Pool(n_processes) as pool:
        results = pool.starmap(
            _load_and_build_single, [(date, min_points, vessel_groups, verbose) for date in all_dates]
        )

    return pd.concat(results, ignore_index=True)
