import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class TSSPLDataset(Dataset):
    """A PyTorch Dataset for vessel trajectory prediction using spatial differences.

    This dataset processes vessel trajectories to create sequences of spatial differences between pairs of
    trajectories, which can be used to train a model to predict vessel movements. Each sequence consists of a
    fixed number of consecutive spatial differences (lat/lon) between two trajectories, with the next
    difference as the target.

    The dataset handles:
    - Calculating spatial differences between trajectory pairs
    - Optional scaling of differences using MinMaxScaler
    - Creating sequences of specified length
    - Converting data to PyTorch tensors

    Attributes:
        sequence_length (int): Number of timesteps in each input sequence
        lat_scaler (MinMaxScaler): Scaler for latitude differences
        lon_scaler (MinMaxScaler): Scaler for longitude differences
        sequences (torch.Tensor): Processed input sequences of shape (n_samples, sequence_length, 2)
        targets (torch.Tensor): Target values of shape (n_samples, 2)
    """

    def __init__(self, trajectories: list[np.ndarray], sequence_length: int = 4, scale_data: bool = True):
        """Initialize the dataset with vessel trajectories.

        Args:
            trajectories: List of trajectory arrays, each with shape (n_points, 4) where each point contains
                        [latitude, longitude, speed, course]. Latitude and longitude are in decimal degrees,
                        speed is in knots, and course is in degrees [0, 360).
            sequence_length: Number of previous timesteps to use for prediction. Default is 4 timesteps.
            scale_data: Whether to scale the spatial differences using MinMaxScaler. Default is True.

        Note:
            The dataset will create sequences from all valid pairs of trajectories.
            A pair is valid if both trajectories are longer than sequence_length + 1.
        """
        if not trajectories:
            raise ValueError("Trajectory list cannot be empty")

        if sequence_length < 1:
            raise ValueError("Sequence length must be positive")

        self.sequence_length = sequence_length
        self.lat_scaler = MinMaxScaler() if scale_data else None
        self.lon_scaler = MinMaxScaler() if scale_data else None

        # Prepare sequences and targets
        X, y = self._prepare_sequences(trajectories)

        if len(X) == 0:
            raise ValueError(
                f"No valid sequences could be created. All trajectories may be too short "
                f"(need length > {sequence_length + 1})"
            )

        self.sequences = torch.FloatTensor(X)
        self.targets = torch.FloatTensor(y)

    def _prepare_sequences(self, trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training sequences from trajectory pairs.

        This method:
        1. Calculates spatial differences between all valid trajectory pairs
        2. Fits scalers on the differences if scaling is enabled
        3. Creates sequences and targets for training

        Args:
            trajectories: List of trajectory arrays, each with shape (n_points, 4) containing [latitude,
                        longitude, speed, course]

        Returns:
            tuple: Contains:
                - X: Array of input sequences with shape (n_samples, sequence_length, 2) where each sequence
                    contains latitude and longitude differences
                - y: Array of target values with shape (n_samples, 2) containing the next latitude and
                    longitude differences for each sequence

        Note:
            Sequences are created using a sliding window approach over the differences between each valid
            trajectory pair. For each window of sequence_length differences, the next difference is used as
            the target.
        """
        # Calculate and store all trajectory pair differences
        trajectory_diffs = []

        for i, traj1 in enumerate(trajectories):
            for j, traj2 in enumerate(trajectories):
                if i != j:
                    min_len = min(len(traj1), len(traj2))
                    if min_len > self.sequence_length + 1:
                        lat_diffs = traj1[:min_len, 0] - traj2[:min_len, 0]
                        lon_diffs = traj1[:min_len, 1] - traj2[:min_len, 1]
                        trajectory_diffs.append((lat_diffs, lon_diffs))

        # Fit scalers if using scaling
        if self.lat_scaler is not None and self.lon_scaler is not None:
            # Stack all lat diffs for fitting
            all_lat_diffs = np.concatenate([diffs[0] for diffs in trajectory_diffs])
            all_lon_diffs = np.concatenate([diffs[1] for diffs in trajectory_diffs])

            self.lat_scaler.fit(all_lat_diffs.reshape(-1, 1))
            self.lon_scaler.fit(all_lon_diffs.reshape(-1, 1))

        # Create sequences for all trajectory pairs
        X_sequences = []
        y_values = []

        for lat_diffs, lon_diffs in trajectory_diffs:
            # Scale the differences if using scaling
            if self.lat_scaler is not None and self.lon_scaler is not None:
                lat_diffs = self.lat_scaler.transform(lat_diffs.reshape(-1, 1)).ravel()
                lon_diffs = self.lon_scaler.transform(lon_diffs.reshape(-1, 1)).ravel()

            # Create sequences from this trajectory pair
            for k in range(len(lat_diffs) - self.sequence_length - 1):
                seq = np.column_stack(
                    (lat_diffs[k : k + self.sequence_length], lon_diffs[k : k + self.sequence_length])
                )
                target = np.array([lat_diffs[k + self.sequence_length], lon_diffs[k + self.sequence_length]])
                X_sequences.append(seq)
                y_values.append(target)

        return np.array(X_sequences), np.array(y_values)

    def get_scalers(self) -> tuple[MinMaxScaler, MinMaxScaler]:
        """Get the fitted scalers for latitude and longitude differences.

        Returns:
            tuple: Contains:
                - lat_scaler: MinMaxScaler fitted on latitude differences
                - lon_scaler: MinMaxScaler fitted on longitude differences
                Both scalers will be None if scale_data was False during initialization.
        """
        return self.lat_scaler, self.lon_scaler

    def __len__(self) -> int:
        """Get the number of sequences in the dataset.

        Returns:
            int: Number of sequences in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its corresponding target.

        Args:
            idx: Index of the sequence to retrieve

        Returns:
            tuple: Contains:
                - sequence: Tensor of shape (sequence_length, 2) containing the input sequence of latitude and
                         longitude differences
                - target: Tensor of shape (2,) containing the target latitude and longitude differences
        """
        return self.sequences[idx], self.targets[idx]
