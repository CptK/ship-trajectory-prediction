"""Module for segmenting AIS data based on status changes."""

from multiprocessing import Pool, cpu_count
from typing import cast
from tqdm import tqdm

import pandas as pd
from shapely.geometry import LineString


def _any_has_type_linestring(df: pd.DataFrame, columns: list[str]) -> bool:
    """Check if any of the columns in a DataFrame has type shapely.geometry.LineString.

    Args:
        df: The DataFrame to check.
        columns: The columns to check.

    Returns:
        True if any of the columns has type shapely.geometry.LineString, False otherwise.
    """
    for column in columns:
        if any(isinstance(value, LineString) for value in df[column]):
            return True

    return False


def _validate_input(
    df: pd.DataFrame,
    status_column: str,
    min_segment_length: int | None,
    additional_list_columns: list[str],
    point_count_column: str | None,
    n_processes: int | None,
) -> None:
    """Validate all input parameters for the segment_by_status function.

    Args:
        df: The DataFrame to segment. Each row contains one trajectory. Must contain the status column, where
            each entry is a list of statuses (int).
        status_column: The name of the column containing the statuses.
        min_segment_length: The minimum length of a segment. If a segment is shorter than this, it will be
            discarded. If None, all segments will be kept.
        additional_list_columns: Additional columns that contain lists that should be segmented in the same
            way as the status column. Type of the elements can either be list or shapely.geometry.LineString.
        point_count_column: The name of the column that should contain the number of points in each segment.
        n_processes: The number of processes to use for parallel processing.

    Raises:
        ValueError: If the status column does not exist in the DataFrame.
        ValueError: If the min_segment_length is not None and not a positive integer.
        ValueError: If any of the additional list columns do not exist in the DataFrame.
        ValueError: If the point_count_column does not exist in the DataFrame and is not None.
        ValueError: If n_processes is not None and not a positive integer.
    """
    if status_column not in df.columns:
        raise ValueError(f"Column '{status_column}' does not exist in the DataFrame.")

    if min_segment_length is not None and min_segment_length <= 0:
        raise ValueError("min_segment_length must be a positive integer or None.")

    for column in additional_list_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    if point_count_column is not None and point_count_column not in df.columns:
        raise ValueError(f"Column '{point_count_column}' does not exist in the DataFrame.")

    if n_processes is not None and n_processes <= 0:
        raise ValueError("n_processes must be a positive integer or None.")


def _get_segments(
    statuses: list[int], split_statuses: list[int], min_segment_length: int | None
) -> list[list[int]]:
    """Get the segments of a trajectory based on the statuses.

    Splits the trajectory at points where the status changes to one of the split statuses. It constructs
    segments between these split points, as the indices of the points between two split points are the
    indices of the segment. If a segment is shorter than the minimum segment length, it will be discarded.
    If there are no split points, the whole trajectory is returned as one segment.

    Args:
        statuses: The statuses of the trajectory.
        split_statuses: The statuses that should be used to split the trajectory.
        min_segment_length: The minimum length of a segment. If a segment is shorter than this, it will be
            discarded. If None, all segments will be kept.

    Returns:
        A list of lists, where each list contains the indices of the points in a segment.
    """
    # Find split points
    split_indices = [i for i, status in enumerate(statuses) if status in split_statuses]

    if not split_indices:
        return [[i for i in range(len(statuses))]]

    # Create segments between split points
    splits = [-1] + sorted(split_indices) + [len(statuses)]
    segments = [
        [i for i in range(splits[j] + 1, splits[j + 1]) if i not in split_indices]
        for j in range(len(splits) - 1)
    ]
    segments = [group for group in segments if group]

    # Filter by minimum length if needed
    if min_segment_length is not None:
        segments = [segment for segment in segments if len(segment) >= min_segment_length]

    return segments


def _process_row(
    row: pd.Series,
    status_column: str,
    default_status: int,
    split_statuses: list[int],
    min_segment_length: int | None,
    additional_list_columns: list[str],
    point_count_column: str | None,
) -> list[pd.Series]:
    """Process a single row and split it into multiple rows based on the status column.

    Args:
        row: The row to process.
        status_column: The name of the column containing the statuses.
        default_status: The default status to use if any status value is None / NaN.
        split_statuses: The statuses that should be used to split the trajectories.
        min_segment_length: The minimum length of a segment. If a segment is shorter than this, it will be
            discarded. If None, all segments will be kept.
        additional_list_columns: Additional columns that contain lists that should be segmented in the same
            way as the status column. Type of the elements can either be list or shapely.geometry.LineString.
        point_count_column: The name of the column that should contain the number of points in each segment.

    Returns:
        A list of rows, where each row contains one segment of the original row.
    """
    # Clean statuses
    statuses = [status if pd.notna(status) else default_status for status in row[status_column]]

    # Get valid segments
    segments = _get_segments(statuses, split_statuses, min_segment_length)

    # Create new rows for each segment
    new_rows = []
    skipped = 0
    for segment in segments:
        try:
            new_row = row.copy()
            new_row[status_column] = [statuses[i] for i in segment]

            for column in additional_list_columns:
                if isinstance(row[column], LineString):
                    new_row[column] = LineString([cast(LineString, row[column]).coords[i] for i in segment])
                else:
                    new_row[column] = [row[column][i] for i in segment]

            if point_count_column is not None:
                new_row[point_count_column] = len(segment)

            new_rows.append(new_row)
        except Exception as e:
            skipped += 1

    return new_rows, skipped


def _process_rows(
    rows: pd.DataFrame,
    status_column: str,
    default_status: int,
    split_statuses: list[int],
    min_segment_length: int | None,
    additional_list_columns: list[str],
    point_count_column: str | None,
) -> pd.DataFrame:
    """Process multiple rows in parallel and split them into multiple rows based on the status column.

    Args:
        rows: The rows to process.
        status_column: The name of the column containing the statuses.
        default_status: The default status to use if any status value is None / NaN.
        split_statuses: The statuses that should be used to split the trajectories.
        min_segment_length: The minimum length of a segment. If a segment is shorter than this, it will be
            discarded. If None, all segments will be kept.
        additional_list_columns: Additional columns that contain lists that should be segmented in the same
            way as the status column. Type of the elements can either be list or shapely.geometry.LineString.
        point_count_column: The name of the column that should contain the number of points in each segment.

    Returns:
        A DataFrame containing the segmented trajectories.
    """
    new_rows = []
    skipped = 0
    for _, row in tqdm(rows.iterrows(), total=len(rows)):
        result, skip = _process_row(
            row=row,
            status_column=status_column,
            default_status=default_status,
            split_statuses=split_statuses,
            min_segment_length=min_segment_length,
            additional_list_columns=additional_list_columns,
            point_count_column=point_count_column,
        )
        new_rows.extend(result)
        skipped += skip
        
    return pd.DataFrame(new_rows), skipped


def _process_in_parallel(df: pd.DataFrame, n_processes: int, **kwargs) -> pd.DataFrame:
    """Process the DataFrame in parallel using multiple processes.

    Args:
        df: The DataFrame to process.
        n_processes: The number of processes to use for parallel processing.
        kwargs: Additional keyword arguments to pass to the _process_rows function.

    Returns:
        A DataFrame containing the segmented trajectories.
    """
    chunk_size = max(1, len(df) // n_processes)
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    with Pool(len(chunks)) as pool:
        results = pool.starmap(
            _process_rows,
            [
                (
                    chunk,
                    kwargs["status_column"],
                    kwargs["default_status"],
                    kwargs["split_statuses"],
                    kwargs["min_segment_length"],
                    kwargs["additional_list_columns"],
                    kwargs["point_count_column"],
                )
                for chunk in chunks
            ],
        )
    dfs = [result[0] for result in results]
    print(f"Skipped {sum(result[1] for result in results)} segments due to errors.")
    return pd.concat(dfs, ignore_index=True).reset_index(drop=True)


def segment_by_status(
    df: pd.DataFrame,
    status_column: str,
    default_status: int,
    split_statuses: list[int],
    min_segment_length: int | None = None,
    additional_list_columns: list[str] = ["geometry", "orientations", "velocities", "timestamps"],
    point_count_column: str | None = "point_count",
    n_processes: int | None = 1,
) -> pd.DataFrame:
    """Segment the given DataFrame by the given status column.

    Args:
        df: The DataFrame to segment. Each row contains one trajectory. Must contain the status column, where
            each entry is a list of statuses (int).
        status_column: The name of the column containing the statuses.
        default_status: The default status to use if any status value is None / NaN.
        split_statuses: The statuses that should be used to split the trajectories.
        min_segment_length: The minimum length of a segment. If a segment is shorter than this, it will be
            discarded. If None, all segments will be kept. If the DataFrame contains LineStrings in any of
            the additional list columns, the minimum segment length will be set to 2 (or kept if already set
            to a higher value) because a LineString with less than two points is not a valid LineString.
        additional_list_columns: Additional columns that contain lists that should be segmented in the same
            way as the status column. Type of the elements can either be list or shapely.geometry.LineString.
        point_count_column: The name of the column that should contain the number of points in each segment.
        n_processes: The number of processes to use for parallel processing. If None, the number of processes
            will be set to the number of CPUs.

    Returns:
        A DataFrame containing the segmented trajectories.

    Raises:
        ValueError: If the status column does not exist in the DataFrame.
        ValueError: If the min_segment_length is not None and not a positive integer.
        ValueError: If any of the additional list columns do not exist in the DataFrame.
        ValueError: If the point_count_column does not exist in the DataFrame and is not None.
        ValueError: If n_processes is not None and not a positive integer.
    """
    _validate_input(
        df=df,
        status_column=status_column,
        min_segment_length=min_segment_length,
        additional_list_columns=additional_list_columns,
        point_count_column=point_count_column,
        n_processes=n_processes,
    )

    if _any_has_type_linestring(df, additional_list_columns):
        if min_segment_length is None or (min_segment_length is not None and min_segment_length < 2):
            min_segment_length = 2

    if n_processes is None:
        n_processes = cpu_count()

    if n_processes == 1:
        return _process_rows(
            rows=df,
            status_column=status_column,
            default_status=default_status,
            split_statuses=split_statuses,
            min_segment_length=min_segment_length,
            additional_list_columns=additional_list_columns,
            point_count_column=point_count_column,
        ).reset_index(drop=True)

    return _process_in_parallel(
        df=df,
        n_processes=n_processes,
        status_column=status_column,
        default_status=default_status,
        split_statuses=split_statuses,
        min_segment_length=min_segment_length,
        additional_list_columns=additional_list_columns,
        point_count_column=point_count_column,
    )
