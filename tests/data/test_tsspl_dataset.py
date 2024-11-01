from unittest import TestCase

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from prediction.data.tsspl_dataset import TSSPLDataset


class TestTSSPLDataset(TestCase):
    """Test cases for the TSSPLDataset class."""

    def setUp(self):
        """Set up test data before each test method."""
        # Create some simple test trajectories
        self.traj1 = np.array(
            [
                [0.0, 0.0, 10.0, 90.0],  # lat, lon, speed, course
                [0.1, 0.1, 10.0, 90.0],
                [0.2, 0.2, 10.0, 90.0],
                [0.3, 0.3, 10.0, 90.0],
                [0.4, 0.4, 10.0, 90.0],
                [0.5, 0.5, 10.0, 90.0],
            ]
        )

        self.traj2 = np.array(
            [
                [0.0, 0.1, 12.0, 85.0],
                [0.1, 0.2, 12.0, 85.0],
                [0.2, 0.3, 12.0, 85.0],
                [0.3, 0.4, 12.0, 85.0],
                [0.4, 0.5, 12.0, 85.0],
            ]
        )

        self.traj3 = np.array(
            [
                [0.1, 0.1, 11.0, 88.0],
                [0.2, 0.2, 11.0, 88.0],
            ]
        )  # Too short to create sequences

        self.trajectories = [self.traj1, self.traj2, self.traj3]

    def test_init_with_valid_data(self):
        """Test dataset initialization with valid trajectory data."""
        dataset = TSSPLDataset(trajectories=self.trajectories, sequence_length=2, scale_data=True)

        self.assertIsInstance(dataset.lat_scaler, MinMaxScaler)
        self.assertIsInstance(dataset.lon_scaler, MinMaxScaler)
        self.assertIsInstance(dataset.sequences, torch.Tensor)
        self.assertIsInstance(dataset.targets, torch.Tensor)

    def test_sequence_creation(self):
        """Test if sequences are created correctly."""
        sequence_length = 2
        dataset = TSSPLDataset(
            trajectories=self.trajectories, sequence_length=sequence_length, scale_data=False
        )

        # Check sequence dimensions
        self.assertEqual(dataset.sequences.dim(), 3)  # (n_samples, seq_len, features)
        self.assertEqual(dataset.sequences.shape[1], sequence_length)
        self.assertEqual(dataset.sequences.shape[2], 2)  # lat and lon differences

        # Check target dimensions
        self.assertEqual(dataset.targets.dim(), 2)  # (n_samples, features)
        self.assertEqual(dataset.targets.shape[1], 2)  # lat and lon differences

        # Check that sequences and targets match in batch dimension
        self.assertEqual(dataset.sequences.shape[0], dataset.targets.shape[0])

    def test_scaling(self):
        """Test if data scaling works correctly."""
        dataset = TSSPLDataset(trajectories=self.trajectories, sequence_length=2, scale_data=True)

        # Check if scaled values are in range [0, 1]
        self.assertTrue(torch.all(dataset.sequences >= -1))
        self.assertTrue(torch.all(dataset.sequences <= 1))
        self.assertTrue(torch.all(dataset.targets >= -1))
        self.assertTrue(torch.all(dataset.targets <= 1))

    def test_no_scaling(self):
        """Test dataset creation without scaling."""
        dataset = TSSPLDataset(trajectories=self.trajectories, sequence_length=2, scale_data=False)

        self.assertIsNone(dataset.lat_scaler)
        self.assertIsNone(dataset.lon_scaler)

    def test_get_scalers(self):
        """Test getting scalers from dataset."""
        dataset = TSSPLDataset(trajectories=self.trajectories, sequence_length=2, scale_data=True)

        lat_scaler, lon_scaler = dataset.get_scalers()
        self.assertIsInstance(lat_scaler, MinMaxScaler)
        self.assertIsInstance(lon_scaler, MinMaxScaler)

    def test_getitem(self):
        """Test if __getitem__ returns correct shapes and types."""
        dataset = TSSPLDataset(trajectories=self.trajectories, sequence_length=2)

        sequence, target = dataset[0]

        self.assertIsInstance(sequence, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
        self.assertEqual(sequence.dim(), 2)  # (seq_len, features)
        self.assertEqual(target.dim(), 1)  # (features,)
        self.assertEqual(sequence.shape[0], 2)  # sequence_length
        self.assertEqual(sequence.shape[1], 2)  # lat and lon differences
        self.assertEqual(target.shape[0], 2)  # lat and lon differences

    def test_dataloader_compatibility(self):
        """Test if dataset works with PyTorch DataLoader."""
        dataset = TSSPLDataset(trajectories=self.trajectories, sequence_length=2)

        batch_size = 2
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Get one batch
        batch_sequences, batch_targets = next(iter(dataloader))

        self.assertEqual(batch_sequences.dim(), 3)  # (batch, seq_len, features)
        self.assertEqual(batch_targets.dim(), 2)  # (batch, features)
        self.assertTrue(batch_sequences.shape[0] <= batch_size)

    def test_empty_trajectories(self):
        """Test handling of empty trajectory list."""
        with self.assertRaises(ValueError):
            TSSPLDataset(trajectories=[], sequence_length=2)

    def test_invalid_sequence_length(self):
        """Test handling of invalid sequence length."""
        with self.assertRaises(ValueError):
            TSSPLDataset(trajectories=self.trajectories, sequence_length=0)

        with self.assertRaises(ValueError):
            TSSPLDataset(trajectories=self.trajectories, sequence_length=-1)

    def test_all_short_trajectories(self):
        """Test handling of case where all trajectories are too short."""
        short_trajectories = [self.traj3]  # Only includes the short trajectory
        with self.assertRaises(ValueError):
            TSSPLDataset(trajectories=short_trajectories, sequence_length=2)
