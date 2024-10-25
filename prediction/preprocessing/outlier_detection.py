"""This module contains functions for detecting and removing outliers in AIS data.

The functionality implemented in this module is based on the following paper:
Zhao L, Shi G, Yang J.
Ship Trajectories Pre-processing Based on AIS Data.
doi:10.1017/S0373463318000188
"""

from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import TypedDict, cast

import numpy as np
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm

from prediction.preprocessing.utils import calc_sog, haversine, timedelta_to_seconds


class BoatType(TypedDict):
    """Type definition for a boat.
    - track (np.ndarray): The track of the boat
    - timestamps (list[datetime]): The timestamps of the track points
    - sogs (list[float]): The speed over ground values of the track points
    """

    track: np.ndarray
    timestamps: list[datetime]
    sogs: list[float]


class SubtrackType(TypedDict):
    tracks: list[np.ndarray]
    timestamps: list[list[datetime]]
    sogs: list[list[float]]


def _partition(
    track: np.ndarray | LineString,
    timestamps: np.ndarray | list[datetime],
    sogs: list[float] | np.ndarray,
    threshold_partition_sog: float = 5.0,
    threshold_partition_distance: float = 50.0,
) -> dict[str, list[np.ndarray | list[datetime] | list[float]]]:
    """This function partitions a track into subtracks based on the speed over ground (SOG) values.

    The partitioning is done by comparing the calculated speed between two points with the given SOG value.
    If the difference between the calculated speed and the SOG value is greater than the threshold, the track
    is partitioned at that point. Additionally, the track is partitioned if the distance between two points is
    greater than the given distance threshold.

    Args:
        track: Track to partition
        timestamps: Timestamps of the track points
        sogs: Speed over ground values of the track points
        threshold_partition_sog: Threshold for partitioning the track based on SOG values
        threshold_partition_distance: Threshold for partitioning the track based on distance between points
                                      in kilometers
    """
    if isinstance(track, LineString):
        track = np.array(track.xy).T

    assert len(track) == len(timestamps) == len(sogs), "Lengths of track, timestamps, and sogs must be equal."

    breakpts = [0]
    for i in range(1, len(sogs)):
        time_delta = timedelta_to_seconds(timestamps[i] - timestamps[i - 1])
        lat1, lng1 = track[i - 1]
        lat2, lng2 = track[i]
        distance = haversine(lat1, lng1, lat2, lng2)
        sog = calc_sog(lat1, lng1, lat2, lng2, time_delta)
        diff = abs(sog - sogs[i])

        if diff > threshold_partition_sog or distance > threshold_partition_distance:
            breakpts.append(i)

    if len(breakpts) == 1:
        return {"tracks": [track], "timestamps": [timestamps], "sogs": [sogs]}

    subtracks: dict[str, list] = {"tracks": [], "timestamps": [], "sogs": []}
    for i in range(len(breakpts)):
        if i == 0:
            start, end = 0, breakpts[i]
        elif i == len(breakpts) - 1:
            start, end = breakpts[i], len(sogs)
        else:
            start, end = breakpts[i], breakpts[i + 1]

        if start != end:
            assert len(track[start:end]) == len(timestamps[start:end]) == len(sogs[start:end])

            subtracks["tracks"].append(track[start:end])
            subtracks["timestamps"].append(timestamps[start:end])
            subtracks["sogs"].append(sogs[start:end])

    assert len(subtracks["tracks"]) == len(subtracks["timestamps"]) == len(subtracks["sogs"])

    return subtracks


def _association(
    subtracks: dict[str, list[np.ndarray | list[datetime] | list[float]]],
    threshold_association_sog: float = 15.0,
    threshold_association_distance: float = 50.0,
    threshold_completeness: int = 100,
):
    """This function associates subtracks based on the speed over ground (SOG) values.

    The association is done by comparing the calculated speed between the last point of one track and the
    first point of another track with the given threshold. If the speed is less than the threshold, the two
    tracks are associated. The association continues until no more tracks can be associated. Finally, the
    associated tracks are checked for completeness and only those with a length greater than the completeness
    threshold are kept. Additionally, the distance between the last point of one track and the first point of
    another track is checked, and the tracks are associated only if the distance is less than the given
    distance threshold.

    Args:
        subtracks: Dictionary containing subtracks, timestamps, and SOG values
        threshold_association_sog: Threshold for associating tracks based on SOG values
        threshold_association_distance: Threshold for associating tracks based on distance between points
                                        in kilometers.
        threshold_completeness: Threshold for the minimum length of associated tracks
    """
    all_tracks: list[np.ndarray] = cast(list[np.ndarray], subtracks["tracks"])
    all_timestamps: list[list[datetime]] = cast(list[list[datetime]], subtracks["timestamps"])
    all_sogs: list[list[float]] = cast(list[list[float]], subtracks["sogs"])

    result: dict[str, list] = {"tracks": [], "timestamps": [], "sogs": []}

    while len(all_tracks) > 1:
        current_track = all_tracks.pop(0)
        current_times = all_timestamps.pop(0)
        current_sogs = all_sogs.pop(0)
        boat: BoatType = {"track": current_track, "timestamps": current_times, "sogs": current_sogs}
        del_indices = []
        for i in range(len(all_tracks)):
            track, timestamps, sogs = all_tracks[i], all_timestamps[i], all_sogs[i]
            lat1, lng1 = boat["track"][-1]
            lat2, lng2 = track[0]
            time_delta = timedelta_to_seconds(timestamps[0] - boat["timestamps"][-1])
            sog = calc_sog(lat1, lng1, lat2, lng2, time_delta)
            distance = haversine(lat1, lng1, lat2, lng2)

            if sog < threshold_association_sog and distance < threshold_association_distance:
                boat["track"] = np.concatenate((boat["track"], track))
                boat["timestamps"].extend(timestamps)
                boat["sogs"].extend(sogs)
                del_indices.append(i)

        if len(boat["track"]) > threshold_completeness:
            result["tracks"].append(boat["track"])
            result["timestamps"].append(boat["timestamps"])
            result["sogs"].append(boat["sogs"])

        for i in sorted(del_indices, reverse=True):
            del all_tracks[i]
            del all_timestamps[i]
            del all_sogs[i]

    return result


def _remove_outliers_row(
    row: pd.Series,
    threshold_partition: float = 5.0,
    threshold_association: float = 15.0,
    threshold_completeness: float = 100.0,
) -> list[pd.Series]:
    """Process a single row of a DataFrame and remove outliers from the track.

    The result is a list of rows, where each row corresponds to a single track after removing outliers. It can
    contain multiple rows if the track is partitioned into multiple subtracks.

    Args:
        row: Input row containing track, timestamps, and SOG values
        threshold_partition: Partition threshold
        threshold_association: Association threshold
        threshold_completeness: Completeness threshold
    """
    track = row["geometry"]
    timestamps = row["timestamps"]
    sogs = row["velocities"]

    subtracks = _partition(track, timestamps, sogs, threshold_partition)
    new_tracks = _association(subtracks, threshold_association, threshold_completeness)

    result = [row.copy() for _ in range(len(new_tracks["tracks"]))]

    for i in range(len(new_tracks["tracks"])):
        result[i]["geometry"] = LineString(new_tracks["tracks"][i])
        result[i]["timestamps"] = new_tracks["timestamps"][i]
        result[i]["velocities"] = new_tracks["sogs"][i]

    return result


def _remove_outliers_chunk(args: tuple) -> list[pd.Series]:
    """
    Process a chunk of the DataFrame in a separate process.

    Args:
        args: Tuple containing (chunk, thresholds, verbose)
    Returns:
        List of processed rows
    """
    chunk, thresholds, verbose = args
    new_rows = []
    iterator = tqdm(chunk.iterrows(), total=len(chunk), disable=not verbose)

    for _, row in iterator:
        rows = _remove_outliers_row(
            row, thresholds["partition"], thresholds["association"], thresholds["completeness"]
        )
        new_rows.extend(rows)

    return new_rows


def remove_outliers_parallel(
    df: pd.DataFrame,
    threshold_partition: float = 5.0,
    threshold_association: float = 15.0,
    threshold_completeness: float = 100.0,
    verbose: bool = True,
    n_processes: int | None = None,
) -> pd.DataFrame:
    """
    Remove outliers from the input DataFrame using multiple processes.

    This function removes outliers from the input DataFrame by partitioning the tracks based on the speed over
    ground (SOG) values and associating the subtracks based on the SOG values. The tracks are then checked for
    completeness and only those with a length greater than the completeness threshold are kept. This algorithm
    follows the methodology described in the paper "Ship Trajectories Pre-processing Based on AIS Data" by
    Zhao L, Shi G, and Yang J.

    Args:
        df: Input DataFrame
        threshold_partition: Partition threshold
        threshold_association: Association threshold
        threshold_completeness: Completeness threshold
        verbose: Whether to show progress bars
        n_processes: Number of processes to use (defaults to CPU count - 1)

    Returns:
        DataFrame with outliers removed
    """
    n_processes = n_processes if n_processes else max(1, cpu_count() - 1)

    if n_processes == 1:
        return remove_outliers(
            df, threshold_partition, threshold_association, threshold_completeness, verbose
        )

    # Calculate chunk size
    chunk_size = len(df) // n_processes
    if chunk_size == 0:
        chunk_size = len(df)

    # Split DataFrame into chunks
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Prepare arguments for each process
    thresholds = {
        "partition": threshold_partition,
        "association": threshold_association,
        "completeness": threshold_completeness,
    }
    process_args = [(chunk, thresholds, verbose) for chunk in chunks]

    # Process chunks in parallel
    with Pool(processes=n_processes) as pool:
        if verbose:
            print(f"Processing with {n_processes} processes...")
        results = pool.map(_remove_outliers_chunk, process_args)

    # Combine results from all processes
    all_rows = []
    for chunk_results in results:
        all_rows.extend(chunk_results)

    return pd.DataFrame(all_rows)


def remove_outliers(
    df: pd.DataFrame,
    threshold_partition: float = 5.0,
    threshold_association: float = 15.0,
    threshold_completeness: float = 100.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Remove outliers from the input DataFrame using a single process.

    This function removes outliers from the input DataFrame by partitioning the tracks based on the speed over
    ground (SOG) values and associating the subtracks based on the SOG values. The tracks are then checked for
    completeness and only those with a length greater than the completeness threshold are kept. This algorithm
    follows the methodology described in the paper "Ship Trajectories Pre-processing Based on AIS Data" by
    Zhao L, Shi G, and Yang J.

    Args:
        df: Input DataFrame
        threshold_partition: Threshold for partitioning the tracks based on SOG values
        threshold_association: Threshold for associating tracks based on SOG values
        threshold_completeness: Threshold for the minimum length of associated tracks
        verbose: Whether to show progress bars
    """
    new_rows = []
    iterator = tqdm(df.iterrows(), total=len(df), disable=not verbose)
    for _, row in iterator:
        rows = _remove_outliers_row(row, threshold_partition, threshold_association, threshold_completeness)
        new_rows.extend(rows)

    return pd.DataFrame(new_rows)
