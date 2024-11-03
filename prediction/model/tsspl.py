"""Trajectory-based Similarity Search Prediction with LSTM (TSSPL) model for vessel trajectory prediction.

This module is based on the TSSPL model presented in the paper:
Vessel Trajectory Prediction Using Historical Automatic Identification System Data (2020) by Alizadeh et al.,
doi: 10.1017/S0373463320000442.
"""

from typing import cast

import numpy as np
import torch
import torch.nn as nn
from fastdtw.fastdtw import fastdtw
from torch.utils.data import DataLoader

from prediction.data.tsspl_dataset import TSSPLDataset
from prediction.preprocessing.utils import dtw_spatial


class SpatialLSTM(nn.Module):
    """Trajectory-based Similarity Search Prediction with LSTM (TSSPL) for vessel trajectory prediction.

    Enhances TSSP by using LSTM neural networks to predict dynamic spatial distances instead of assuming
    constant distances. Combines DTW for trajectory similarity matching with LSTM for learning complex
    patterns in distance variations. Accounts for maritime environmental factors and vessel dynamics. Most
    accurate of the three models, especially for long-term predictions. Requires substantial historical AIS
    data for LSTM training and trajectory matching. Best suited for applications requiring high-accuracy
    predictions in dynamic maritime environments.

    Example:
    >>> import matplotlib.pyplot as plt
    >>> from numpy import array
    >>> from numpy.random import normal, randint
    >>> from prediction.model.tsspl import TSSPL
    >>> db = [array(
    ...     [[48 + i * 0.1 + normal(0, 0.01), -125 + 0.1 * i + normal(0, 0.01), 15, 180] for i in range(12)]
    ... ) for _ in range(2)]

    >>> target = array([[48.1+i*0.1+normal(0,0.01), -125.1+0.1*i+ normal(0,0.01), 15, 180] for i in range(5)])
    >>> tsspl = TSSPL(db, n_epochs=2)
    >>> predictions, similar_traj = tsspl.predict(target, n_steps=5)

    >>> plt.figure(figsize=(10, 6))
    >>> plt.plot(target[:, 1], target[:, 0], "b-o", label="Target")
    >>> plt.plot(similar_traj[:, 1], similar_traj[:, 0], "g-o", label="Most Similar")
    >>> plt.plot(predictions[:, 1], predictions[:, 0], "r--o", label="Predictions")
    >>> plt.xlabel("Longitude")
    >>> plt.ylabel("Latitude")
    >>> plt.legend()
    >>> plt.show()
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 1):
        """LSTM model for predicting spatial differences."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=2,  # lat_diff, lon_diff
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, 2)
               where 2 represents [lat_diff, lon_diff]

        Returns:
            torch.Tensor: Predicted differences of shape (batch_size, 2)

        Raises:
            ValueError: If input tensor has wrong shape or dimensions
        """
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch_size, sequence_length, features), " f"got shape {x.shape}"
            )

        if x.shape[-1] != 2:
            raise ValueError(
                f"Expected 2 features (lat_diff, lon_diff) in last dimension, " f"got {x.shape[-1]}"
            )

        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return cast(torch.Tensor, predictions)


class TSSPL:
    def __init__(
        self,
        trajectory_database: list[np.ndarray],
        w_d: float = 0.6,
        w_s: float = 0.2,
        w_c: float = 0.2,
        hidden_size: int = 64,
        sequence_length: int = 4,
        learning_rate: float = 0.01,
        n_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize TSSPL with database, weights and LSTM parameters."""
        if not np.isclose(w_d + w_s + w_c, 1.0):
            raise ValueError("Weights must sum to 1.0")
        if any(w < 0 for w in [w_d, w_s, w_c]):
            raise ValueError("Weights must be non-negative")

        self.database = trajectory_database
        self.w_d = w_d
        self.w_s = w_s
        self.w_c = w_c
        self.sequence_length = sequence_length
        self.device = device

        # Create dataset and get scalers
        self.dataset = TSSPLDataset(trajectory_database, sequence_length)
        self.lat_scaler, self.lon_scaler = self.dataset.get_scalers()

        # Initialize and train LSTM model
        self.lstm_model = SpatialLSTM(hidden_size=hidden_size)
        self.lstm_model.to(device)
        self._train_lstm(epochs=n_epochs, lr=learning_rate)

    def _train_lstm(self, batch_size: int = 32, epochs: int = 10, lr: float = 0.01):
        """Train LSTM model on trajectory data."""
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=lr)

        self.lstm_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.lstm_model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def _find_most_similar_trajectory(self, target_traj: np.ndarray) -> np.ndarray:
        """Find most similar trajectory from database using DTW."""
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
            d_course_norm = d_course / 180.0

            total_dist = self.w_d * d_spatial_norm + self.w_s * d_speed_norm + self.w_c * d_course_norm

            if total_dist < min_dist:
                min_dist = total_dist
                most_similar = hist_traj

        return cast(np.ndarray, most_similar)

    def predict(self, target_traj: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Predict future locations for target trajectory."""
        similar_traj = self._find_most_similar_trajectory(target_traj)

        # Get recent spatial differences
        recent_lat_diffs = (
            target_traj[-self.sequence_length :, 0]
            - similar_traj[len(target_traj) - self.sequence_length : len(target_traj), 0]
        )
        recent_lon_diffs = (
            target_traj[-self.sequence_length :, 1]
            - similar_traj[len(target_traj) - self.sequence_length : len(target_traj), 1]
        )

        # Scale differences
        recent_lat_diffs = self.lat_scaler.transform(recent_lat_diffs.reshape(-1, 1)).ravel()
        recent_lon_diffs = self.lon_scaler.transform(recent_lon_diffs.reshape(-1, 1)).ravel()

        # Prepare LSTM input
        lstm_input = (
            torch.FloatTensor(np.column_stack((recent_lat_diffs, recent_lon_diffs)))
            .unsqueeze(0)
            .to(self.device)
        )

        # Generate predictions
        predictions = np.zeros((n_steps, 2))
        self.lstm_model.eval()

        with torch.no_grad():
            current_sequence = lstm_input

            for i in range(n_steps):
                # Predict next spatial differences
                pred_diffs = self.lstm_model(current_sequence)
                pred_diffs = pred_diffs.cpu().numpy()

                # Inverse transform predictions
                pred_lat_diff = self.lat_scaler.inverse_transform(pred_diffs[:, 0].reshape(-1, 1))
                pred_lon_diff = self.lon_scaler.inverse_transform(pred_diffs[:, 1].reshape(-1, 1))

                # Calculate predicted position
                next_idx = len(target_traj) + i
                if next_idx < len(similar_traj):
                    pred_lat = similar_traj[next_idx, 0] + pred_lat_diff[0, 0]
                    pred_lon = similar_traj[next_idx, 1] + pred_lon_diff[0, 0]
                    predictions[i] = [pred_lat, pred_lon]

                    # Update sequence for next prediction
                    new_diffs = torch.FloatTensor([[pred_diffs[0, 0], pred_diffs[0, 1]]])
                    current_sequence = torch.cat([current_sequence[:, 1:, :], new_diffs.unsqueeze(1)], dim=1)
                else:
                    print(f"Similar trajectory too short. Could only generate {i} of {n_steps} predictions.")
                    break

        return predictions, similar_traj
