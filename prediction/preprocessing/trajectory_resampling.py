from multiprocessing import Pool

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastdtw.fastdtw import fastdtw
from shapely.geometry import LineString, Point
from tqdm.auto import tqdm

from prediction.preprocessing.utils import haversine


def _interpolate_point_on_line(line: LineString, distance: float) -> Point:
    """Interpolate a point on a LineString at a given distance from the start.

    If the distance is less than 0, the start point is returned, if the distance is greater than the length
    of the line, the end point is returned. Otherwise, the point is interpolated along the line.

    Args:
        line (LineString): A LineString object.
        distance (float): The distance along the line to interpolate a point.

    Returns:
        Point: A new Point object at the interpolated position.
    """
    if distance <= 0.0:
        return Point(list(line.coords)[0])
    if distance >= line.length:
        return Point(list(line.coords)[-1])
    return Point(line.interpolate(distance))


def _interpolate_circular(start_angle: float, end_angle: float, fraction: float) -> float:
    """Interpolate between two angles (in degrees) along the shortest path.

    The interpolation is based on the fraction between the two angles, where 0 is the start angle and 1 is the
    end angle. The interpolation is done along the shortest path between the two angles, which means that it
    will correctly interpolate across the 0/360 degree boundary.

    Args:
        start_angle (float): The start angle in degrees.
        end_angle (float): The end angle in degrees.
        fraction (float): The interpolation fraction between 0 and 1.

    Returns:
        float: The interpolated angle in degrees.

    Examples:
        >>> interpolate_circular(0, 180, 0.5)
        90.0
        >>> interpolate_circular(350, 10, 0.25)
        355.0
    """
    start_angle = start_angle % 360
    end_angle = end_angle % 360
    diff = (end_angle - start_angle + 180) % 360 - 180
    return (start_angle + diff * fraction) % 360


def _process_single_trajectory(row: pd.Series, interval_seconds: int = 300) -> pd.Series:
    """Process a single trajectory row.

    This function resamples a single trajectory to a fixed time interval. The trajectory is assumed to be a
    LineString with associated timestamps, velocities, orientations, and statuses. The resampling is done by
    interpolating the position, velocity, and orientation at each new timestamp.

    Args:
        row (pd.Series): A single row from a trajectory DataFrame.
        interval_seconds (int): The time interval in seconds to resample to.

    Returns:
        pd.Series: A new row with the resampled trajectory.
    """

    line = row["geometry"]
    times = row["timestamps"]
    vels = row["velocities"]
    orients = row["orientations"]
    stats = row["statuses"]

    # Convert coords to list once at the start
    coords = list(line.coords)

    # Start with the first point
    start_time = times[0]

    # Calculate end time and round to nearest interval
    end_time = times[-1]
    remaining_time = (end_time - start_time) % interval_seconds
    if remaining_time > interval_seconds / 2:
        end_time = end_time + (interval_seconds - remaining_time)
    else:
        end_time = end_time - remaining_time

    # Generate new timestamps
    new_times = list(range(start_time, end_time + 1, interval_seconds))

    # Initialize result lists
    new_points = []
    new_velocities = []
    new_orientations = []
    new_statuses = []

    # Process each new timestamp
    for new_time in new_times:
        if new_time == times[0]:
            # Keep first point as is
            new_points.append(Point(coords[0]))
            new_velocities.append(vels[0])
            new_orientations.append(orients[0])
            new_statuses.append(stats[0])
            continue

        # Find bracketing points
        next_idx = np.searchsorted(times, new_time)
        if next_idx >= len(times):
            break
        prev_idx = next_idx - 1

        # Calculate interpolation fraction based on time
        time_range = times[next_idx] - times[prev_idx]
        time_fraction = (new_time - times[prev_idx]) / time_range

        # Calculate the distance along the line for this time fraction
        prev_point = Point(coords[prev_idx])
        next_point = Point(coords[next_idx])
        start_dist = line.project(prev_point)
        end_dist = line.project(next_point)
        current_dist = start_dist + (end_dist - start_dist) * time_fraction

        # Interpolate position along the line
        new_point = _interpolate_point_on_line(line, current_dist)
        new_points.append(new_point)

        # Interpolate velocity
        new_vel = vels[prev_idx] + (vels[next_idx] - vels[prev_idx]) * time_fraction
        new_velocities.append(new_vel)

        # Interpolate orientation
        new_orient = _interpolate_circular(orients[prev_idx], orients[next_idx], time_fraction)
        new_orientations.append(new_orient)

        # Forward fill status
        new_statuses.append(stats[prev_idx])

    # Create new LineString from points
    new_line = LineString([(p.x, p.y) for p in new_points])
    new_row = row.copy()
    new_row["geometry"] = new_line
    new_row["timestamps"] = new_times
    new_row["velocities"] = new_velocities
    new_row["orientations"] = new_orientations
    new_row["statuses"] = new_statuses

    return new_row


def resample_trajectories(df: pd.DataFrame, interval_minutes: int = 5) -> pd.DataFrame:
    """Resample trajectory data to fixed time intervals.

    This function resamples a DataFrame of trajectory data to a fixed time interval. The DataFrame is assumed
    to have columns for geometry (LineString), timestamps (list of integers), velocities (list of floats),
    orientations (list of floats), and statuses (list of integers). The resampling is done by interpolating
    the position, velocity, and orientation at each new timestamp.

    Args:
        df (pd.DataFrame): A DataFrame with trajectory data.
        interval_minutes (int): The time interval in minutes to resample to.

    Returns:
        pd.DataFrame: A new DataFrame with the resampled trajectories

    Examples:
        >>> from prediction.preprocessing.trajectory_resampling import resample_trajectories
        >>> from prediction.preprocessing.trajectory_resampling import create_sample_trajectory
        >>> from prediction.preprocessing.trajectory_resampling import plot_trajectories
        >>> import matplotlib.pyplot as plt
        >>> original_df = create_sample_trajectory()
        >>> resampled_df = resample_trajectories(original_df, interval_minutes=5)
        >>> fig = plot_trajectories(original_df, resampled_df)
        >>> plt.show()
        >>> print("Original timestamps (minutes):", [t//60 for t in original_df.iloc[0]['timestamps']])
        >>> print("Resampled timestamps (minutes):", [t//60 for t in resampled_df.iloc[0]['timestamps']])
    """
    interval_seconds = interval_minutes * 60

    # Process each trajectory
    results: list[pd.Series] = []
    for _, row in df.iterrows():
        results.append(_process_single_trajectory(row, interval_seconds))

    return pd.DataFrame(results)


def create_sample_trajectory() -> pd.DataFrame:  # pragma: no cover
    """Create a sample trajectory DataFrame for testing the resampling function."""
    # Create a more complex trajectory showing various movement patterns
    t = np.array([0, 2, 3, 7, 8, 15, 16, 19, 25, 28])  # irregular times in minutes
    t = t * 60  # convert to seconds

    # Create coordinates (with some sharp turns and varying speeds)
    x = np.array([0.0, 0.2, 0.4, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    y = np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0, 0.5])

    # Create sample velocities, orientations, and statuses
    velocities = np.linspace(5, 15, len(t)) + np.random.normal(0, 1, len(t))
    orientations = np.linspace(0, 360, len(t)) % 360  # Simple increasing orientation
    statuses = np.ones(len(t), dtype=int)  # all "under way" for simplicity

    # Create LineString
    line = LineString(zip(x, y))

    # Create DataFrame
    df = pd.DataFrame(
        {
            "geometry": [line],
            "timestamps": [t.tolist()],
            "velocities": [velocities.tolist()],
            "orientations": [orientations.tolist()],
            "statuses": [statuses.tolist()],
        }
    )

    return df


def plot_trajectories(
    original_df: pd.DataFrame, resampled_df: pd.DataFrame
) -> plt.Figure:  # pragma: no cover
    """Plot original and resampled trajectories for comparison.

    This function creates a plot showing the original and resampled trajectories side by side. The plot
    includes the original and resampled points, as well as time labels for each point.

    Args:
        original_df (pd.DataFrame): The original trajectory DataFrame.
        resampled_df (pd.DataFrame): The resampled trajectory DataFrame.

    Returns:
        plt.Figure: The resulting matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot original trajectory
    original_line = original_df.iloc[0]["geometry"]
    original_times = original_df.iloc[0]["timestamps"]
    x_orig, y_orig = zip(*original_line.coords)

    ax.plot(x_orig, y_orig, "b-", label="Original Trajectory", alpha=0.6)
    ax.scatter(x_orig, y_orig, c="blue", s=50, label="Original Points")

    # Add time labels for original points
    for i, (x, y, t) in enumerate(zip(x_orig, y_orig, original_times)):
        ax.annotate(
            f"{t//60}min", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8, color="blue"
        )

    # Plot resampled trajectory
    resampled_line = resampled_df.iloc[0]["geometry"]
    resampled_times = resampled_df.iloc[0]["timestamps"]
    x_resampled, y_resampled = zip(*resampled_line.coords)

    ax.plot(x_resampled, y_resampled, "r--", label="Resampled Trajectory", alpha=0.6)
    ax.scatter(x_resampled, y_resampled, c="red", s=50, label="Resampled Points")

    # Add time labels for resampled points
    for x, y, t in zip(x_resampled, y_resampled, resampled_times):
        ax.annotate(
            f"{t//60}min", (x, y), xytext=(5, -10), textcoords="offset points", fontsize=8, color="red"
        )

    ax.grid(True)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Original vs Resampled Trajectory\n(labels show time in minutes)")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    return fig


def _linestring_to_coords(linestring: LineString) -> list[tuple[float, float]]:
    """Extract coordinates from a LineString as (lat, lon) pairs.

    Args:
        linestring: A LineString object.

    Returns:
        List of coordinate pairs as [(lat1, lon1), (lat2, lon2), ...]
    """
    coords = list(linestring.coords)
    return [(y, x) for x, y in coords]  # Convert from (lon, lat) to (lat, lon)


def _calculate_dtw_distance(
    coords1: list[tuple[float, float]], coords2: list[tuple[float, float]]
) -> tuple[float, float]:
    """Calculate DTW distance between two coordinate lists for total DTW distance and average point distance.

    This function uses FastDTW algorithm (an approximation of DTW with linear time complexity) to find
    the optimal alignment between two trajectories. Distances between points are calculated using
    the haversine formula, which gives the great-circle distance between two points on a sphere
    using their latitudes and longitudes.

    Args:
        coords1: List of coordinate pairs (latitude, longitude) for the first trajectory
        coords2: List of coordinate pairs (latitude, longitude) for the second trajectory

    Returns:
        tuple[float, float]: A tuple containing:
            - Total DTW distance in kilometers: The cumulative haversine distance of the optimal path
            - Average point distance in kilometers: The total DTW distance divided by the length of the
              longer trajectory

    Note:
        - Input coordinates should be in decimal degrees (e.g., (29.15, -90.21))
        - Uses fastdtw implementation which provides O(n) time complexity
        - Uses haversine formula for calculating distances between points
        - Coordinates must be within valid ranges: latitude [-90, 90], longitude [-180, 180]
        - Earth radius of 6371 km is used in distance calculations
    """
    distance, _ = fastdtw(coords1, coords2, dist=lambda x, y: haversine(x[0], x[1], y[0], y[1]))

    # Calculate average distance per matched point pair
    avg_point_distance = distance / max(len(coords1), len(coords2))

    return distance, avg_point_distance


def compare_trajectory_pairs(
    df1: gpd.GeoDataFrame | pd.DataFrame,
    df2: gpd.GeoDataFrame | pd.DataFrame,
    geometry_col: str = "geometry",
    n_processes: int = 1,
) -> pd.DataFrame:
    """Compare similarity of trajectory pairs using DTW distance and length ratio.

    This function compares pairs of trajectories from two DataFrames to quantify their similarity.
    It calculates Dynamic Time Warping (DTW) distance and path length ratios between the original
    and resampled trajectories. The function supports parallel processing for large datasets.

    Args:
        df1: First GeoDataFrame containing original LineString geometries. Must contain 'MMSI' column
            and geometry column specified by geometry_col parameter.
        df2: Second GeoDataFrame containing resampled LineString geometries. Must contain 'MMSI' column
            and geometry column specified by geometry_col parameter.
        geometry_col: Name of the geometry column containing LineString objects. Defaults to "geometry".
        n_processes: Number of parallel processes to use. If > 1, data will be split into batches
            and processed in parallel. Defaults to 1 (sequential processing).

    Returns:
        pd.DataFrame: DataFrame containing similarity metrics for each trajectory pair:
            - mmsi: Maritime Mobile Service Identity number
            - trajectory_id: Index of the trajectory pair
            - dtw_distance: Total DTW distance in kilometers
            - avg_point_distance: Average point-to-point distance in kilometers
            - path_length_ratio: Absolute difference in number of points divided by max length
            - original_points: Number of points in original trajectory
            - resampled_points: Number of points in resampled trajectory

    Note:
        - DataFrames must be aligned and have the same length
        - MMSI values must match between corresponding rows
        - Geometries are converted to coordinate lists before comparison
        - DTW distances are calculated using haversine formula
        - Progress bar is shown for sequential processing
        - For parallel processing, data is split into approximately equal batches

    Raises:
        ValueError: If lengths of the two DataFrames do not match
        AssertionError: If MMSI values don't match between corresponding trajectories
    """
    if len(df1) != len(df2):
        raise ValueError("DataFrames must have the same length")

    if n_processes > 1:
        batch_size = len(df1) // n_processes
        batches_1 = [df1.iloc[i : i + batch_size] for i in range(0, len(df1), batch_size)]
        batches_2 = [df2.iloc[i : i + batch_size] for i in range(0, len(df2), batch_size)]

        with Pool(n_processes) as pool:
            results = pool.starmap(compare_trajectory_pairs, zip(batches_1, batches_2))
        df = pd.concat(results)

        # sort by trajectory_id
        df.sort_values("trajectory_id", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # Get common indices
    results = []

    # Use tqdm for progress bar
    for idx in tqdm(range(len(df1))):
        # Get geometries
        geom1 = df1.iloc[idx][geometry_col]
        geom2 = df2.iloc[idx][geometry_col]

        mmsi1 = df1.iloc[idx]["MMSI"]
        mmsi2 = df2.iloc[idx]["MMSI"]

        assert mmsi1 == mmsi2, f"MMSI mismatch: {mmsi1} != {mmsi2}"

        # Convert to coordinate lists
        coords1 = _linestring_to_coords(geom1)
        coords2 = _linestring_to_coords(geom2)

        # Calculate DTW distance
        dtw_distance, avg_point_distance = _calculate_dtw_distance(coords1, coords2)

        # Calculate length ratio
        path_length_ratio = abs(len(coords1) - len(coords2)) / max(len(coords1), len(coords2))

        results.append(
            {
                "mmsi": mmsi1,
                "trajectory_id": df1.index[idx],
                "dtw_distance": dtw_distance,
                "avg_point_distance": avg_point_distance,
                "path_length_ratio": path_length_ratio,
                "original_points": len(coords1),
                "resampled_points": len(coords2),
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Create and process sample data
    original_df = create_sample_trajectory()
    resampled_df = resample_trajectories(original_df, interval_minutes=5)

    # Create plot
    fig = plot_trajectories(original_df, resampled_df)
    plt.show()

    # Print some information about the trajectories
    print("\nOriginal timestamps (minutes):", [t // 60 for t in original_df.iloc[0]["timestamps"]])
    print("Resampled timestamps (minutes):", [t // 60 for t in resampled_df.iloc[0]["timestamps"]])
