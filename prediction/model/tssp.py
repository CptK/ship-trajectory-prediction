from typing import cast

import numpy as np
from fastdtw.fastdtw import fastdtw

from prediction.preprocessing.utils import dtw_spatial


class TSSP:
    def __init__(
        self, trajectory_database: list[np.ndarray], w_d: float = 0.6, w_s: float = 0.2, w_c: float = 0.2
    ):
        """Initialize TSSP with database and weights.

        Args:
            trajectory_database: List of trajectory arrays, each with shape (n_points, 4)
                               containing [lat, lon, sog, cog] where:
                               - lat: Latitude in decimal degrees
                               - lon: Longitude in decimal degrees
                               - sog: Speed over ground in knots
                               - cog: Course over ground in degrees [0, 360)
            w_d: Weight for spatial distance (0-1)
            w_s: Weight for speed similarity (0-1)
            w_c: Weight for course similarity (0-1)

        Note:
            Weights should sum to 1.0
        """
        self.database = trajectory_database
        self.w_d = w_d
        self.w_s = w_s
        self.w_c = w_c

    def predict(self, target_traj: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Predict future locations for target trajectory.

        Args:
            target_traj: Target trajectory array with shape (n_points, 4)
                        containing [lat, lon, sog, cog] where:
                        - lat: Latitude in decimal degrees
                        - lon: Longitude in decimal degrees
                        - sog: Speed over ground in knots
                        - cog: Course over ground in degrees [0, 360)
            n_steps: Number of future steps to predict

        Returns:
            tuple: (predictions, similar_traj)
                - predictions: Array of predicted locations with shape (n_steps, 2) containing [lat, lon]
                - similar_traj: Most similar trajectory array from database with same format as input
        """
        similar_traj = self._find_most_similar_trajectory(target_traj)
        predictions = self._predict_locations(target_traj, similar_traj, n_steps)
        return predictions, similar_traj

    def _find_most_similar_trajectory(self, target_traj: np.ndarray) -> np.ndarray:
        """Find most similar trajectory from database using DTW.

        Args:
            target_traj: Target trajectory array with shape (n_points, 4)
                        containing [lat, lon, sog, cog] where:
                        - lat: Latitude in decimal degrees
                        - lon: Longitude in decimal degrees
                        - sog: Speed over ground in knots
                        - cog: Course over ground in degrees [0, 360)

        Returns:
            Most similar trajectory array from database with same format as input

        Example:
            >>> target = np.array([[48.5, -125.5, 12.5, 180.0],
                                 [48.6, -125.4, 12.3, 182.0]])
            >>> similar_traj = tssp.find_most_similar_trajectory(target)
        """
        min_dist = float("inf")
        most_similar = None

        for hist_traj in self.database:
            # Spatial DTW
            d_spatial, _ = dtw_spatial(target_traj, hist_traj)

            # Speed DTW
            d_speed, _ = fastdtw(
                target_traj[:, 2], hist_traj[:, 2], dist=lambda x, y: abs(float(x) - float(y))
            )

            # Course DTW
            d_course, _ = fastdtw(
                target_traj[:, 3],
                hist_traj[:, 3],
                dist=lambda x, y: min(abs(float(x) - float(y)), 360 - abs(float(x) - float(y))),
            )

            # Normalize distances
            d_spatial_norm = d_spatial / (np.max([len(target_traj), len(hist_traj)]))
            d_speed_norm = d_speed / (np.max([target_traj[:, 2].max(), hist_traj[:, 2].max()]))
            d_course_norm = d_course / 180.0  # max possible course difference is 180

            total_dist = self.w_d * d_spatial_norm + self.w_s * d_speed_norm + self.w_c * d_course_norm

            if total_dist < min_dist:
                min_dist = total_dist
                most_similar = hist_traj

        return cast(np.ndarray, most_similar)

    def _predict_locations(
        self, target_traj: np.ndarray, similar_traj: np.ndarray, n_steps: int
    ) -> np.ndarray:
        """
        Predict future locations using constant spatial distances.

        Args:
            target_traj: Current trajectory array with shape (n_points, 4)
                        containing [lat, lon, sog, cog]
            similar_traj: Most similar trajectory array with same format
            n_steps: Number of future steps to predict

        Returns:
            Array of predicted locations with shape (n_steps, 2) containing [lat, lon]

        Example:
            >>> predictions = tssp.predict_locations(target_traj, similar_traj, n_steps=6)
            >>> print(predictions.shape)  # (6, 2)
        """
        lat_diff = target_traj[-1, 0] - similar_traj[len(target_traj) - 1, 0]
        lon_diff = target_traj[-1, 1] - similar_traj[len(target_traj) - 1, 1]

        start_idx = len(target_traj)
        end_idx = min(start_idx + n_steps, len(similar_traj))

        if end_idx - start_idx < n_steps:
            print(f"Similar trajectory is too short. Could only generate {end_idx - start_idx} of {n_steps}.")

        future_points = similar_traj[start_idx:end_idx, :2].copy()
        future_points[:, 0] += lat_diff
        future_points[:, 1] += lon_diff

        return future_points


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate sample data
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

    # Create database of trajectories
    database = [generate_sample_trajectory(48.0, -125.0, points=np.random.randint(8, 15)) for _ in range(20)]

    # Create target trajectory
    target_traj = generate_sample_trajectory(48.1, -125.1, points=5)

    # Initialize TSSP
    tssp = TSSP(database)

    # Find most similar trajectory and predict future positions
    predictions, similar_traj = tssp.predict(target_traj, n_steps=5)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(target_traj[:, 1], target_traj[:, 0], "b-o", label="Target")
    plt.plot(similar_traj[:, 1], similar_traj[:, 0], "g-o", label="Most Similar")
    plt.plot(predictions[:, 1], predictions[:, 0], "r--o", label="Predictions")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.title("TSSP Trajectory Prediction")
    plt.show()
