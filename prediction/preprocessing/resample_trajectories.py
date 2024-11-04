import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


def interpolate_point_on_line(line: LineString, distance: float) -> Point:
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


def interpolate_circular(start_angle: float, end_angle: float, fraction: float) -> float:
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


def process_single_trajectory(row: pd.Series, interval_seconds: int = 300) -> pd.Series:
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
        new_point = interpolate_point_on_line(line, current_dist)
        new_points.append(new_point)

        # Interpolate velocity
        new_vel = vels[prev_idx] + (vels[next_idx] - vels[prev_idx]) * time_fraction
        new_velocities.append(new_vel)

        # Interpolate orientation
        new_orient = interpolate_circular(orients[prev_idx], orients[next_idx], time_fraction)
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
        >>> from prediction.preprocessing.resample_trajectories import resample_trajectories
        >>> from prediction.preprocessing.resample_trajectories import create_sample_trajectory
        >>> from prediction.preprocessing.resample_trajectories import plot_trajectories
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
        results.append(process_single_trajectory(row, interval_seconds))

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
