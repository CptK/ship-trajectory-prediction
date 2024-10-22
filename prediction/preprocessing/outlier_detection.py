import numpy as np
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count


def timedelta_to_seconds(delta: timedelta):
    total_seconds = delta.total_seconds()
    total_microseconds = delta.microseconds
    return total_seconds + (total_microseconds / 1000000)

def extract_times(time: datetime):
    return time.hour, time.minute, time.second

def calc_speed(lat1, lng1, lat2, lng2, time):
    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1
    a = (np.sin(delta_lat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(delta_lng / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = 6371 * c # haversine distance
    return distance / (time + 1e-10) # add small number to avoid division by zero


def partition(
    track: np.ndarray | LineString,
    timestamps: np.ndarray | list[datetime],
    sogs: list[float] | np.ndarray,
    threshold_partition: float = 5.0,
) -> dict[str, list[np.ndarray | list[datetime] | list[float]]]:
    if isinstance(track, LineString):
        track = np.array(track.xy).T

    assert len(track) == len(timestamps) == len(sogs), "Lengths of track, timestamps, and sogs must be equal."

    breakpts = [0]
    for i in range(1, len(sogs)):
        time_delta = timedelta_to_seconds(timestamps[i] - timestamps[i - 1])
        lat1, lng1 = track[i - 1]
        lat2, lng2 = track[i]
        speed = calc_speed(lat1, lng1, lat2, lng2, time_delta)
        diff = abs(speed - sogs[i])

        if diff > threshold_partition:
            breakpts.append(i)

    if len(breakpts) == 1:
        return {"tracks": [track], "timestamps": [timestamps], "sogs": [sogs]}

    subtracks = {"tracks": [], "timestamps": [], "sogs": []}
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


def association(
    subtracks: dict[str, list[np.ndarray | list[datetime] | list[float]]],
    threshold_association: float = 15.0,
    threshold_completeness: int = 100,
):
    all_tracks, all_timestamps, all_sogs = subtracks["tracks"], subtracks["timestamps"], subtracks["sogs"]

    result = {"tracks": [], "timestamps": [], "sogs": []}

    while len(all_tracks) > 1:
        current_track, current_timestamps, current_sogs = all_tracks.pop(0), all_timestamps.pop(0), all_sogs.pop(0)
        boat = {"track": current_track, "timestamps": current_timestamps, "sogs": current_sogs}
        del_indices = []
        for i in range(len(all_tracks)):
            track, timestamps, sogs = all_tracks[i], all_timestamps[i], all_sogs[i]
            lat1, lng1 = boat["track"][-1]
            lat2, lng2 = track[0]
            time_delta = timedelta_to_seconds(timestamps[0] - boat["timestamps"][-1])
            speed = calc_speed(lat1, lng1, lat2, lng2, time_delta)

            if speed < threshold_association:
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


def remove_outliers_row(
    row: pd.Series,
    threshold_partition: float = 5.0,
    threshold_association: float = 15.0,
    threshold_completeness: float = 100.0,
) -> list[pd.Series]:
    track = row["geometry"]
    timestamps = row["timestamps"]
    sogs = row["velocities"]

    subtracks = partition(track, timestamps, sogs, threshold_partition)
    new_tracks = association(subtracks, threshold_association, threshold_completeness)

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
        rows = remove_outliers_row(
            row,
            thresholds['partition'],
            thresholds['association'],
            thresholds['completeness']
        )
        new_rows.extend(rows)
    
    return new_rows


def remove_outliers_parallel(
    df: pd.DataFrame,
    threshold_partition: float = 5.0,
    threshold_association: float = 15.0,
    threshold_completeness: float = 100.0,
    verbose: bool = True,
    n_processes: int = None,
) -> pd.DataFrame:
    """
    Parallel version of remove_outliers function.
    
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
    
    # Calculate chunk size
    chunk_size = len(df) // n_processes
    if chunk_size == 0:
        chunk_size = len(df)
    
    # Split DataFrame into chunks
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Prepare arguments for each process
    thresholds = {
        'partition': threshold_partition,
        'association': threshold_association,
        'completeness': threshold_completeness
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
    new_rows = []
    iterator = tqdm(df.iterrows(), total=len(df), disable=not verbose)
    for _, row in iterator:
        rows = remove_outliers_row(row, threshold_partition, threshold_association, threshold_completeness)
        new_rows.extend(rows)

    return pd.DataFrame(new_rows)
