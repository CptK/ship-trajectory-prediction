import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class AISDataSet(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame,
        n_pred: int,
        seq_len: int | None = None,
        include_speed: bool = True,
        include_orientation: bool = True,
        include_timestamps: bool = True,
        n_workers: int | None = None,
        verbose: bool = True
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
        self.X = []
        self.y = []
        self.n_pred = n_pred
        self.seq_len = seq_len
        self.include_speed = include_speed
        self.include_orientation = include_orientation
        self.include_timestamps = include_timestamps
        self.n_workers = n_workers if n_workers else max(cpu_count() - 1, 1)
        self.verbose = verbose

        self._process_df(df)

    def _process_row(self, row: pd.Series):
        n_points = len(row["velocities"])
        train_start_idx = 0 if not self.seq_len else n_points - self.seq_len - self.n_pred
        train_end_idx = n_points - self.n_pred

        geometry = np.array(row["geometry"].xy).T
        start_time = row["timestamps"][0]
        timestamps = [(t - start_time).total_seconds() for t in row["timestamps"]]

        x_geometry = geometry[train_start_idx:train_end_idx]
        x_velocities = row["velocities"][train_start_idx:train_end_idx]
        x_timestatmps = timestamps[train_start_idx:train_end_idx]
        x_orientations = row["orientations"][train_start_idx:train_end_idx]

        y_geometry = geometry[train_end_idx:]
        y_velocities = row["velocities"][train_end_idx:]
        y_timestatmps = timestamps[train_end_idx:]
        y_orientations = row["orientations"][train_end_idx:]

        X = x_geometry
        y = y_geometry

        if self.include_speed:
            # right now x has shape (n_points, 2), adding speed will make it (n_points, 3)
            X = np.concatenate([X, np.array(x_velocities)[:, None]], axis=1)
            y = np.concatenate([y, np.array(y_velocities)[:, None]], axis=1)

        if self.include_orientation:
            X = np.concatenate([X, np.array(x_orientations)[:, None]], axis=1)
            y = np.concatenate([y, np.array(y_orientations)[:, None]], axis=1)

        if self.include_timestamps:
            X = np.concatenate([X, np.array(x_timestatmps)[:, None]], axis=1)
            y = np.concatenate([y, np.array(y_timestatmps)[:, None]], axis=1)

        return X, y

    def _process_chunk(self, df: pd.DataFrame):
        X, Y = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), disable=not self.verbose):
            x, y = self._process_row(row)
            X.append(x)
            Y.append(y)

        return X, Y

    def _process_df(self, df: pd.DataFrame):
        chunk_size = len(df) // self.n_workers
        if chunk_size == 0:
            chunk_size = len(df)

        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        with Pool(self.n_workers) as p:
            results = p.map(self._process_chunk, chunks)

        for x, y in results:
            self.X.extend(x)
            self.y.extend(y)

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
