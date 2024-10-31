from typing import cast

import numpy as np

from prediction.preprocessing.utils import azimuth, haversine


class PSSP:
    def __init__(
        self,
        trajectory_database: list[np.ndarray],
        w_d: float = 0.6,
        w_s: float = 0.2,
        w_c: float = 0.2,
        azimuth_threshold: float = 22.5,
    ):
        """Initialize PSSP with database and weights.

        Args:
            trajectory_database: List of trajectory arrays, where each trajectory
                               is a ndarray with shape (n_points, n_features)
                               First 4 features must be [lat, lon, sog, cog]:
                               - lat: Latitude in decimal degrees
                               - lon: Longitude in decimal degrees
                               - sog: Speed over ground in knots
                               - cog: Course over ground in degrees [0, 360)
                               Additional features are allowed but ignored
            w_d: Weight for spatial distance (0-1)
            w_s: Weight for speed similarity (0-1)
            w_c: Weight for course similarity (0-1)
            azimuth_threshold: Maximum allowed azimuth difference in degrees

        Note:
            Weights should sum to 1.0
        """
        if not trajectory_database:
            raise ValueError("Database cannot be empty")

        self.database = trajectory_database
        self.w_d = w_d
        self.w_s = w_s
        self.w_c = w_c
        self.azimuth_threshold = azimuth_threshold

    def predict(self, target_point: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Predict future locations for target point.

        Args:
            target_point: Target point array with shape (4,)
                         containing [lat, lon, sog, cog]
            n_steps: Number of future steps to predict

        Returns:
            tuple: (predictions, similar_points)
                - predictions: Array of predicted locations, shape (n_steps, 2) containing [lat, lon]
                - similar_points: Array of similar points used for predictions, shape (n_steps, n_features)
        """
        if target_point.shape[0] < 4:
            raise ValueError("Target point must have at least [lat, lon, sog, cog]")

        predictions = np.zeros((n_steps, 2))
        similar_points = np.zeros((n_steps, self.database[0].shape[1]))

        current_target = target_point.copy()

        for i in range(n_steps):
            # Find most similar point and its trajectory index
            similar_point, traj_idx, point_idx = self._find_most_similar_point(current_target)
            if similar_point is None:
                print(f"Could only generate {i} of {n_steps} predictions.")
                return predictions[:i], similar_points[:i]

            similar_points[i] = similar_point

            # Get next point from the same trajectory
            next_point = self._get_next_point(traj_idx, point_idx)
            if next_point is None:
                print(f"Could only generate {i} of {n_steps} predictions.")
                return predictions[:i], similar_points[:i]

            # Calculate spatial differences
            lat_diff = current_target[0] - similar_point[0]
            lon_diff = current_target[1] - similar_point[1]

            # Predict next position
            predictions[i, 0] = next_point[0] + lat_diff
            predictions[i, 1] = next_point[1] + lon_diff

            # Update target for next iteration
            current_target = np.array(
                [
                    predictions[i, 0],
                    predictions[i, 1],
                    next_point[2],  # Use similar trajectory's SOG
                    next_point[3],  # Use similar trajectory's COG
                ]
            )

        return predictions, similar_points

    def _find_most_similar_point(
        self, target_point: np.ndarray
    ) -> tuple[np.ndarray | None, int | None, int | None]:
        """Find most similar point from database using weighted similarity measures.

        Returns:
            Tuple of (similar_point, trajectory_index, point_index)
        """
        min_dist = float("inf")
        most_similar = None
        best_traj_idx = None
        best_point_idx = None

        for traj_idx, trajectory in enumerate(self.database):
            n_points = len(trajectory)
            if n_points < 2:  # Skip trajectories that are too short
                continue

            for point_idx in range(n_points - 1):  # -1 to ensure we can get next point
                hist_point = trajectory[point_idx]
                next_hist = trajectory[point_idx + 1]

                # Check azimuth constraint
                target_azimuth = azimuth(target_point[0], target_point[1], next_hist[0], next_hist[1])
                hist_azimuth = azimuth(hist_point[0], hist_point[1], next_hist[0], next_hist[1])
                if abs(target_azimuth - hist_azimuth) > self.azimuth_threshold:
                    continue

                # Calculate similarities
                d_spatial = haversine(target_point[0], target_point[1], hist_point[0], hist_point[1])
                d_speed = abs(float(target_point[2] - hist_point[2]))
                d_course = min(
                    abs(float(target_point[3] - hist_point[3])),
                    360 - abs(float(target_point[3] - hist_point[3])),
                )

                # Normalize distances
                d_spatial_norm = d_spatial / 100.0
                d_speed_norm = d_speed / 30.0
                d_course_norm = d_course / 180.0

                # Calculate total distance
                total_dist = self.w_d * d_spatial_norm + self.w_s * d_speed_norm + self.w_c * d_course_norm

                if total_dist < min_dist:
                    min_dist = total_dist
                    most_similar = hist_point
                    best_traj_idx = traj_idx
                    best_point_idx = point_idx

        return most_similar, best_traj_idx, best_point_idx

    def _get_next_point(self, traj_idx: int | None, point_idx: int | None) -> np.ndarray | None:
        """Get the next point from a trajectory given indices."""
        if traj_idx is None or point_idx is None:
            return None

        trajectory = self.database[traj_idx]
        if point_idx + 1 >= len(trajectory):
            return None

        return cast(np.ndarray, trajectory[point_idx + 1])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def generate_sample_trajectory(start_lat, start_lon, points=10, noise=0.01):
        """Generate noisy trajectory data."""
        trajectory = np.zeros((points, 4))
        for i in range(points):
            trajectory[i] = [
                start_lat + i * 0.1 + np.random.normal(0, noise),  # lat
                start_lon + i * 0.1 + np.random.normal(0, noise),  # lon
                15 + np.random.normal(0, 1),  # speed
                180 + np.random.normal(0, 5),  # course
            ]
        return trajectory

    # Generate database with slightly different starting points
    database = [
        generate_sample_trajectory(
            start_lat=48.0 + np.random.normal(0, 0.3), start_lon=-125.0 + np.random.normal(0, 0.3), points=20
        )
        for _ in range(10)
    ]

    # Create target point (similar to trajectories but slightly offset)
    target = np.array([48.1, -125.1, 15.0, 45.0, 0.0])

    # Initialize PSSP
    pssp = PSSP(database)

    # Predict future positions
    predictions, similar_points = pssp.predict(target, n_steps=5)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot all trajectories
    for traj in database:
        plt.plot(traj[:, 1], traj[:, 0], "gray", alpha=0.3)

    # Plot target, similar points and predictions
    plt.plot(target[1], target[0], "b*", markersize=15, label="Target")
    if len(similar_points) > 0:
        plt.plot(similar_points[:, 1], similar_points[:, 0], "g-o", label="Similar Points", markersize=10)
        plt.plot(predictions[:, 1], predictions[:, 0], "r--o", label="Predictions", markersize=8)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.title("PSSP Point Prediction")
    plt.show()
