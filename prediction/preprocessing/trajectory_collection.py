from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

import pandas as pd
from shapely.geometry import LineString, Point
from tqdm import tqdm

import os
from pathlib import Path

from prediction.data.loader import load_data

from shapely.wkt import dumps as wkt_dumps
from shapely.geometry import LineString, Point
from shapely.validation import make_valid
from shapely.wkt import loads as wkt_loads


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


def load_and_build(
    start_date: datetime,
    end_date: datetime,
    min_points: int,
    vessel_groups: dict[int, str],
    n_processes: int = None,
    verbose: bool = True
) -> pd.DataFrame:
    n_processes = n_processes if n_processes is not None else max(cpu_count() - 1, 1)
    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

    if verbose:
        print(f"Loading and building trajectories for {len(all_dates)} days using {n_processes} processes")

    with Pool(n_processes) as pool:
        results = pool.starmap(
            _load_or_build_for_date,
            [(date, min_points, vessel_groups, verbose) for date in all_dates]
        )

    return pd.concat(results, ignore_index=True)


def _load_or_build_for_date(
    date: datetime,
    min_points: int,
    vessel_groups: dict[int, str],
    verbose: bool = True
) -> pd.DataFrame:
    """Check if data exists for the given date, load or build trajectories."""
    save_path = Path(__file__).resolve().parents[2] / "data" / "trajectories"
    filename = f'trajectories{date.strftime("%Y%m%d")}.parquet'
    
    if os.path.exists(save_path / filename) and os.path.getsize(save_path / filename) > 0:
        if verbose:
            print(f"Loading existing data for {date}")
        # Load the data from the Parquet file
        trajectories = pd.read_parquet(save_path / filename)
        
        # Saved data not in useable format
        # Convert WKT back to geometries
        trajectories['geometry'] = trajectories['geometry'].apply(lambda wkt: wkt_loads(wkt) if isinstance(wkt, str) else wkt)
        # Convert timestamps to Python datetime objects
        trajectories['timestamps'] = trajectories['timestamps'].apply(lambda ts_list: [pd.to_datetime(ts) for ts in ts_list])

        return trajectories
    
    else:
        # If the file doesn't exist, build the trajectories and save them
        if verbose:
            print(f"Building new data for {date}")
        df = load_data(date)
        trajectories = build_trajectories(df, min_points, vessel_groups, verbose)
        # linestring cannot be saved as parquet
        #Convert to WKT in another dataframe
        trajectories_save = trajectories.copy()
        trajectories_save['geometry'] = trajectories['geometry'].apply(lambda geom: wkt_dumps(geom) if geom else None)
        trajectories_save.to_parquet(save_path / filename, index=False)
        
        return trajectories

def validate_lengths(trajectories: pd.DataFrame, verbose: bool = True) -> None:
    """Validate that the lengths of timestamps, sogs, and geometry coordinates are equal.

    Args:
        trajectories: The DataFrame containing built trajectories.
        verbose: Whether to print validation results.
    """
    mismatches = []

    for idx, row in trajectories.iterrows():
        if row['geometry'] is not None:
            geometry_length = len(row['geometry'].coords)
        else:
            geometry_length = 0

        timestamps_length = len(row['timestamps']) if row['timestamps'] is not None else 0
        sogs_length = len(row['velocities']) if row['velocities'] is not None else 0

        if geometry_length != timestamps_length or timestamps_length != sogs_length:
            mismatches.append({
                'MMSI': row['mmsi'],
                'Date': row['start_time'].date() if row['start_time'] else None,
                'geometry_length': geometry_length,
                'timestamps_length': timestamps_length,
                'sogs_length': sogs_length
            })

    if verbose:
        if mismatches:
            print("Length mismatches found in the following trajectories:")
            for mismatch in mismatches:
                print(f"MMSI: {mismatch['MMSI']}, Date: {mismatch['Date']}, "
                      f"geometry length: {mismatch['geometry_length']}, "
                      f"timestamps length: {mismatch['timestamps_length']}, "
                      f"sogs length: {mismatch['sogs_length']}")
        else:
            print("All trajectories have matching lengths for geometry, timestamps, and sogs.")

def save_data(df: pd.DataFrame, file_path: Path) -> None:
    """
    Convert geometry column to WKT format and save the DataFrame to a parquet file.
    """
    df_copy = df.copy()
    df_copy['geometry'] = df_copy['geometry'].apply(lambda geom: wkt_dumps(geom) if geom else None)
    df_copy.to_parquet(file_path, index=False)

#def load_data


