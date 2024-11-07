import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class AISDataSet(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        n_pred: int,
        seq_len: int | None = None,
        normalize: bool = True,
        include_speed: bool = True,
        include_orientation: bool = True,
        include_timestamps: bool = True,
        n_workers: int | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            df: DataFrame containing the AIS data. Following columns are expected:
                - geometry: shapely.geometry.LineString
                - velocities: List[float]
                - orientations: List[float]
                - timestamps: List[datetime.datetime]
            n_pred: Number of points to predict.
            seq_len: Number of points to consider before predicting.
            include_speed: Whether to include speed in the input and output.
            include_orientation: Whether to include orientation in the input and output.
            include_timestamps: Whether to include timestamps in the input and output.
            n_workers: Number of workers to use for processing the data.
            verbose: Whether to show progress bars.
        """
        self.X: list = []
        self.y: list = []
        self.n_pred = n_pred
        self.seq_len = seq_len
        self.normalize = normalize
        self.include_speed = include_speed
        self.include_orientation = include_orientation
        self.include_timestamps = include_timestamps
        self.n_workers = n_workers if n_workers else max(cpu_count() - 1, 1)
        self.verbose = verbose

        self._process_df(df)

        if self.normalize:
            self.X, self.y = self._normalize(self.X, self.y)

    def _process_row(self, row: pd.Series):
        n_points = len(row["velocities"])
        train_start_idx = 0 if not self.seq_len else n_points - self.seq_len - self.n_pred
        train_end_idx = n_points - self.n_pred

        geometry = np.array(row["geometry"].xy).T

        if len(geometry) < self.seq_len + self.n_pred:
            return None, None

        x_res = geometry[train_start_idx:train_end_idx]
        y_res = geometry[train_end_idx:]

        if self.include_speed:
            # right now x has shape (n_points, 2), adding speed will make it (n_points, 3)
            x_velocities = row["velocities"][train_start_idx:train_end_idx]
            y_velocities = row["velocities"][train_end_idx:]

            x_res = np.concatenate([x_res, np.array(x_velocities)[:, None]], axis=1)
            y_res = np.concatenate([y_res, np.array(y_velocities)[:, None]], axis=1)

        if self.include_orientation:
            x_orientations = row["orientations"][train_start_idx:train_end_idx]
            y_orientations = row["orientations"][train_end_idx:]

            x_res = np.concatenate([x_res, np.array(x_orientations)[:, None]], axis=1)
            y_res = np.concatenate([y_res, np.array(y_orientations)[:, None]], axis=1)

        if self.include_timestamps:
            start_time = row["timestamps"][0]
            timestamps = [(t - start_time).total_seconds() for t in row["timestamps"]]
            x_timestatmps = timestamps[train_start_idx:train_end_idx]
            y_timestatmps = timestamps[train_end_idx:]

            x_res = np.concatenate([x_res, np.array(x_timestatmps)[:, None]], axis=1)
            y_res = np.concatenate([y_res, np.array(y_timestatmps)[:, None]], axis=1)

        return x_res, y_res

    def _process_chunk(self, df: pd.DataFrame):
        x_res, y_res = [], []
        invalid_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), disable=not self.verbose):
            x, y = self._process_row(row)

            if x is not None and y is not None:
                x_res.append(x)
                y_res.append(y)
            else:
                invalid_count += 1

        return x_res, y_res, invalid_count

    def _process_df(self, df: pd.DataFrame):
        chunk_size = len(df) // self.n_workers
        if chunk_size == 0:
            chunk_size = len(df)

        chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

        with Pool(self.n_workers) as p:
            results = p.map(self._process_chunk, chunks)

        invalid_count = 0
        for x, y, count in results:
            self.X.extend(x)
            self.y.extend(y)
            invalid_count += count

        if invalid_count > 0:
            print(f"Skipped {invalid_count} rows due to insufficient data")

    def _normalize(self, X: list, y: list):  # noqa: N803
        normalized_x = []
        normalized_y = []

        # Process each sequence individually
        for seq_x, seq_y in zip(X, y):
            # Convert to numpy if not already
            seq_x = np.array(seq_x)
            seq_y = np.array(seq_y)

            # Lat/Lon normalization
            seq_x[:, :2] /= np.array([90, 180])
            seq_y[:, :2] /= np.array([90, 180])

            current_idx = 2

            # Speed normalization
            if self.include_speed:
                max_speed = 30.0  # TODO: make this a parameter
                seq_x[:, current_idx] = np.clip(seq_x[:, current_idx], 0, max_speed) / max_speed * 2 - 1
                seq_y[:, current_idx] = np.clip(seq_y[:, current_idx], 0, max_speed) / max_speed * 2 - 1
                current_idx += 1

            # Course normalization
            if self.include_orientation:
                angles_x = np.deg2rad(seq_x[:, current_idx])
                angles_y = np.deg2rad(seq_y[:, current_idx])

                sin_cos_x = np.column_stack([np.sin(angles_x), np.cos(angles_x)])
                sin_cos_y = np.column_stack([np.sin(angles_y), np.cos(angles_y)])

                seq_x = np.column_stack(
                    [
                        seq_x[:, :current_idx],
                        sin_cos_x,
                        (
                            seq_x[:, current_idx + 1 :]
                            if current_idx + 1 < seq_x.shape[1]
                            else np.array([]).reshape(seq_x.shape[0], 0)
                        ),
                    ]
                )

                seq_y = np.column_stack(
                    [
                        seq_y[:, :current_idx],
                        sin_cos_y,
                        (
                            seq_y[:, current_idx + 1 :]
                            if current_idx + 1 < seq_y.shape[1]
                            else np.array([]).reshape(seq_y.shape[0], 0)
                        ),
                    ]
                )

                current_idx += 2

            # Timestamp normalization
            if self.include_timestamps:
                seq_x[:, current_idx] = np.log1p(seq_x[:, current_idx]) / np.log(24 * 3600)
                seq_y[:, current_idx] = np.log1p(seq_y[:, current_idx]) / np.log(24 * 3600)

            normalized_x.append(seq_x)
            normalized_y.append(seq_y)

        return normalized_x, normalized_y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Get a single trajectory sample.

        Returns:
            tuple: (X, y) where:
                X: array of shape (seq_len, n_features)
                y: array of shape (n_pred, n_features)
                n_features depends on included features:
                - 2 (lat, lon) +
                - 1 if include_speed +
                - 1 if include_orientation +
                - 1 if include_timestamps
        """
        return self.X[idx], self.y[idx]


class AISCompletionDataset(AISDataSet):
    """Dataset for trajectory completion task with seq2seq models.

    Every sample consists of a 3-tuple:
    - input_seq: Array of shape (seq_len, n_features)
    - decoder_input: Array of shape (n_pred, n_features), the first element is the last element of input_seq,
        the rest are the target_seq without the last element
    - target_seq: Array of shape (n_pred, n_features)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        n_pred: int,
        **kwargs,
    ) -> None:
        """Initialize the trajectory completion dataset.

        Args:
            df: DataFrame containing the AIS data
            seq_len: Number of points to consider before predicting
            n_pred: Number of points to predict
            **kwargs: Additional arguments passed to AISDataSet
        """
        super().__init__(df, n_pred=n_pred, seq_len=seq_len, **kwargs)

    def __getitem__(self, idx):
        """Get a trajectory completion sample.

        Returns:
            tuple: (input_seq, target_seq) where:
                input_seq: array (seq_len, n_features) with:
                    - Lat/Lon normalized to [-1, 1]
                    - Speed normalized to [-1, 1]
                    - Course normalized to sin/cos
                    - Timestamp normalized to [0, 1]
                target_seq: array (n_pred, n_features) with the same normalization
        """
        X, y = super().__getitem__(idx)
        y_all = np.concatenate([y[-1][None], y], axis=0)

        # create decoder_input as y without the last element
        decoder_input = y_all[:-1]

        return X, decoder_input, y


class MaskedAISDataset(AISDataSet):
    def __init__(
        self,
        df: pd.DataFrame,
        max_seq_len: int,
        random_mask_ratio: float = 0.15,
        consecutive_mask_ratio: float = 0.8,
        end_mask_ratio: float = 0.5,
        mask_token: float = -1.0,
        min_masks: int = 1,
        pad_token: float = -999.0,
        # Strategy probabilities (must sum to 1)
        random_mask_prob: float = 0.2,
        consecutive_mask_prob: float = 0.3,
        end_mask_prob: float = 0.5,
        **kwargs,
    ) -> None:
        """Initialize the masked trajectory dataset.

        Args:
            df: DataFrame containing the AIS data
            max_seq_len: Maximum sequence length (for padding)
            mask_ratio: Percentage of points to mask
            mask_token: Value to use for masking
            min_masks: Minimum number of points to mask per sequence
            pad_token: Value to use for padding
            random_mask_prob: Probability of using random masking strategy
            consecutive_mask_prob: Probability of using consecutive masking strategy
            end_mask_prob: Probability of using end masking strategy
            **kwargs: Additional arguments passed to AISDataSet

        Raises:
            ValueError: If probabilities don't sum to 1 (within tolerance)
            ValueError: If any probability is negative
        """
        super().__init__(df, n_pred=0, seq_len=None, **kwargs)

        self.max_seq_len = max_seq_len
        self.random_mask_ratio = random_mask_ratio
        self.consecutive_mask_ratio = consecutive_mask_ratio
        self.end_mask_ratio = end_mask_ratio
        self.mask_token = mask_token
        self.min_masks = min_masks
        self.pad_token = pad_token

        # Validate probabilities
        probs = [random_mask_prob, consecutive_mask_prob, end_mask_prob]
        if not all(p >= 0 for p in probs):
            raise ValueError("All probabilities must be non-negative")
        if not np.isclose(sum(probs), 1.0):
            raise ValueError("Probabilities must sum to 1")

        self.strategy_probs = probs

    def _create_random_masks(self, seq_len: int) -> np.ndarray:
        """Create masks by randomly selecting points."""
        n_masks = max(self.min_masks, int(seq_len * self.random_mask_ratio))
        mask = np.zeros(seq_len, dtype=bool)
        mask_positions = random.sample(range(seq_len), n_masks)
        mask[mask_positions] = True
        return mask

    def _create_consecutive_masks(self, seq_len: int) -> np.ndarray:
        """Create masks for consecutive points."""
        n_masks = max(self.min_masks, int(seq_len * self.consecutive_mask_ratio))
        mask = np.zeros(seq_len, dtype=bool)
        start_idx = random.randint(0, seq_len - n_masks)
        mask[start_idx : start_idx + n_masks] = True
        return mask

    def _create_end_masks(self, seq_len: int) -> np.ndarray:
        """Create masks for the final portion of the actual sequence.

        For a sequence length of 50 and mask_ratio of 0.2:
        [0 0 0 ... 0 1 1 ... 1]
        |__________|_______|
            40 real    10 masked
        (Padding to fill up max_seq_len will be added later)

        Args:
            seq_len: Length of the actual sequence (before padding)

        Returns:
            np.ndarray: Boolean mask where True indicates positions to mask
        """
        n_masks = max(self.min_masks, int(seq_len * self.end_mask_ratio))
        mask = np.zeros(seq_len, dtype=bool)
        mask[-n_masks:] = True  # Mask last n_masks positions of actual sequence
        return mask

    def _create_masks(self, seq_len: int) -> np.ndarray:
        """Create masks using randomly selected strategy."""
        # Choose strategy based on probabilities
        strategy_idx = np.random.choice(3, p=self.strategy_probs)

        if strategy_idx == 0:
            return self._create_random_masks(seq_len)
        elif strategy_idx == 1:
            return self._create_consecutive_masks(seq_len)
        else:
            return self._create_end_masks(seq_len)

    def _pad_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Pad sequence to max_seq_len."""
        pad_length = self.max_seq_len - len(seq)
        if pad_length <= 0:
            return seq[: self.max_seq_len]

        pad_shape = (pad_length, seq.shape[1])
        padding = np.full(pad_shape, self.pad_token)
        return np.vstack([seq, padding])

    def __getitem__(self, idx):
        """Get a masked trajectory sample.

        For example, with:
        - max_seq_len = 100
        - actual sequence length = 50
        - mask_ratio = 0.2

        Returns:
            tuple: (masked_input, attention_mask, target) where:
                masked_input: array (max_seq_len, n_features) with:
                    - Original values for unmasked positions
                    - mask_token (-1.0) for masked positions
                    - pad_token (-999.0) for padding
                attention_mask: boolean array (max_seq_len,) where:
                    - True for real data (masked or unmasked)
                    - False for padding
                target: array (max_seq_len, n_features) with:
                    - Original values for all positions
                    - pad_token (-999.0) for padding
        """
        # Get original sequence
        seq, _ = super().__getitem__(idx)

        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len :]

        # Create mask for actual sequence (before padding)
        orig_len = len(seq)
        mask = self._create_masks(orig_len)

        # First pad the original sequence
        padded_seq = self._pad_sequence(seq)
        attention_mask = np.arange(self.max_seq_len) < orig_len

        # Apply masking only to the actual sequence portion
        masked_seq = padded_seq.copy()
        expanded_mask = np.tile(mask[:, None], (1, padded_seq.shape[1]))
        padded_expanded_mask = np.zeros((self.max_seq_len, padded_seq.shape[1]), dtype=bool)
        padded_expanded_mask[:orig_len] = expanded_mask

        # Apply mask token to masked positions in actual sequence
        masked_seq[padded_expanded_mask] = self.mask_token

        return (masked_seq.astype(np.float32), attention_mask, padded_seq.astype(np.float32))
