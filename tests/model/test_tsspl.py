from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from prediction.data.tsspl_dataset import TSSPLDataset
from prediction.model.tsspl import TSSPL, SpatialLSTM


class TestSpatialLSTM(TestCase):
    """Test cases for the SpatialLSTM model."""

    def setUp(self):
        """Set up test cases."""
        self.batch_size = 32
        self.sequence_length = 4
        self.hidden_size = 64
        self.num_layers = 2
        self.input_size = 2  # lat_diff, lon_diff
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        self.model = SpatialLSTM(hidden_size=self.hidden_size, num_layers=self.num_layers).to(self.device)

        # Create sample input
        self.sample_input = torch.randn(self.batch_size, self.sequence_length, self.input_size).to(
            self.device
        )

    def test_model_initialization(self):
        """Test if model initializes with correct architecture."""
        # Check model attributes
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.num_layers, self.num_layers)

        # Check LSTM configuration
        self.assertEqual(self.model.lstm.input_size, self.input_size)
        self.assertEqual(self.model.lstm.hidden_size, self.hidden_size)
        self.assertEqual(self.model.lstm.num_layers, self.num_layers)

        # Check fully connected layer
        self.assertEqual(self.model.fc.in_features, self.hidden_size)
        self.assertEqual(self.model.fc.out_features, 2)

    def test_forward_shape(self):
        """Test if forward pass produces correct output shape."""
        output = self.model(self.sample_input)

        expected_shape = (self.batch_size, 2)  # (batch_size, lat_diff + lon_diff)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_batch_first(self):
        """Test if model handles batch_first input correctly."""
        # Try different batch sizes
        batch_sizes = [1, 16, 32]
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, self.sequence_length, self.input_size).to(self.device)
            output = self.model(input_tensor)
            self.assertEqual(output.shape, (batch_size, 2))

    def test_sequence_length_invariance(self):
        """Test if model can handle different sequence lengths."""
        sequence_lengths = [2, 4, 8]
        for seq_len in sequence_lengths:
            input_tensor = torch.randn(self.batch_size, seq_len, self.input_size).to(self.device)
            output = self.model(input_tensor)
            self.assertEqual(output.shape, (self.batch_size, 2))

    def test_training_step(self):
        """Test if model can perform a training step."""
        # Create sample data
        input_tensor = self.sample_input
        target = torch.randn(self.batch_size, 2).to(self.device)

        # Set up optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.MSELoss()

        # Perform training step
        self.model.train()
        optimizer.zero_grad()
        output = self.model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Check if loss is valid
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_gradient_flow(self):
        """Test if gradients flow through the model properly."""
        input_tensor = self.sample_input
        target = torch.randn(self.batch_size, 2).to(self.device)

        # Forward pass
        output = self.model(input_tensor)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        # Check if gradients exist and are not zero
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.allclose(param.grad, torch.zeros_like(param.grad)))  # type: ignore

    def test_model_eval_mode(self):
        """Test if model behaves correctly in eval mode."""
        self.model.eval()
        with torch.no_grad():
            output1 = self.model(self.sample_input)
            output2 = self.model(self.sample_input)

        # Outputs should be identical in eval mode with same input
        self.assertTrue(torch.allclose(output1, output2))

    def test_device_movement(self):
        """Test if model can be moved between devices."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Move model to CPU
        model_cpu = self.model.cpu()
        input_cpu = self.sample_input.cpu()
        output_cpu = model_cpu(input_cpu)

        # Move model to CUDA
        model_cuda = self.model.cuda()
        input_cuda = self.sample_input.cuda()
        output_cuda = model_cuda(input_cuda)

        # Check outputs have correct device
        self.assertEqual(output_cpu.device.type, "cpu")
        self.assertEqual(output_cuda.device.type, "cuda")

    def test_parameter_count(self):
        """Test if model has expected number of parameters."""
        # Print actual parameter shapes and counts
        total_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()

        # Calculate expected parameters with detailed breakdown
        # First layer
        lstm_layer1_weights = 4 * (self.input_size * self.hidden_size)  # ih weights
        lstm_layer1_hidden = 4 * (self.hidden_size * self.hidden_size)  # hh weights
        lstm_layer1_bias = 4 * 2 * self.hidden_size  # ih and hh biases

        # Second layer
        lstm_layer2_weights = 4 * (self.hidden_size * self.hidden_size)  # ih weights
        lstm_layer2_hidden = 4 * (self.hidden_size * self.hidden_size)  # hh weights
        lstm_layer2_bias = 4 * 2 * self.hidden_size  # ih and hh biases

        # FC layer
        fc_weights = self.hidden_size * 2  # hidden_size * output_features
        fc_bias = 2  # output_features bias

        lstm_params = (
            lstm_layer1_weights
            + lstm_layer1_hidden
            + lstm_layer1_bias
            + lstm_layer2_weights
            + lstm_layer2_hidden
            + lstm_layer2_bias
        )
        fc_params = fc_weights + fc_bias
        expected_total = lstm_params + fc_params

        self.assertEqual(total_params, expected_total)

    def test_invalid_input_handling(self):
        """Test model's handling of invalid inputs."""
        # Test with wrong input size (features)
        with self.assertRaises(ValueError) as context:
            invalid_input = torch.randn(self.batch_size, self.sequence_length, 3).to(self.device)
            self.model(invalid_input)
        self.assertIn("Expected 2 features", str(context.exception))

        # Test with wrong input dimensions
        with self.assertRaises(ValueError) as context:
            invalid_input = torch.randn(self.batch_size, self.input_size).to(self.device)
            self.model(invalid_input)
        self.assertIn("Expected 3D input tensor", str(context.exception))


class TestTSSPL(TestCase):
    """Test cases for the TSSPL model."""

    def setUp(self):
        """Set up test data."""
        # Create sample trajectories
        self.traj1 = np.array([[48.0, -125.0 - i * 0.1, 4.0, 270.0] for i in range(20)])
        self.traj2 = np.array([[48.0 + i * 0.1, -125.0, 12.0, 0.0] for i in range(18)])

        # Create proper list of trajectories
        self.trajectories = [self.traj1, self.traj2]
        self.sequence_length = 2
        self.hidden_size = 32

        # Initialize model
        self.model = TSSPL(
            trajectory_database=self.trajectories,
            sequence_length=self.sequence_length,
            hidden_size=self.hidden_size,
            learning_rate=0.005,
            n_epochs=15,
        )

    def test_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model.dataset, TSSPLDataset)
        self.assertIsInstance(self.model.lat_scaler, MinMaxScaler)
        self.assertIsInstance(self.model.lon_scaler, MinMaxScaler)
        self.assertEqual(self.model.sequence_length, self.sequence_length)
        self.assertEqual(self.model.lstm_model.hidden_size, self.hidden_size)

        # Check weights sum to 1
        self.assertAlmostEqual(self.model.w_d + self.model.w_s + self.model.w_c, 1.0)

    def test_find_most_similar_trajectory(self):
        """Test finding most similar trajectory with different weights."""
        # Create trajectories with distinct characteristics
        traj_slow_west = np.array([[48.0, -125.0 - i * 0.1, 8.0, 270.0] for i in range(10)])  # Slow, westward
        traj_fast_north = np.array(
            [[48.0 + i * 0.1, -125.0, 20.0, 0.0] for i in range(10)]
        )  # Fast, northward
        traj_med_east = np.array(
            [[48.0, -125.0 + i * 0.1, 15.0, 90.0] for i in range(10)]
        )  # Medium, eastward

        # Create models with different weights
        model_spatial = TSSPL([traj_slow_west, traj_fast_north, traj_med_east], w_d=1.0, w_s=0.0, w_c=0.0)
        model_speed = TSSPL([traj_slow_west, traj_fast_north, traj_med_east], w_d=0.0, w_s=1.0, w_c=0.0)
        model_course = TSSPL([traj_slow_west, traj_fast_north, traj_med_east], w_d=0.0, w_s=0.0, w_c=1.0)

        # Test trajectory moving west at medium speed
        target = np.array([[48.0, -125.0 - i * 0.1, 15.0, 270.0] for i in range(5)])

        # Get similar trajectories using different weights
        similar_spatial = model_spatial._find_most_similar_trajectory(target)
        similar_speed = model_speed._find_most_similar_trajectory(target)
        similar_course = model_course._find_most_similar_trajectory(target)

        # Basic checks
        self.assertIsInstance(similar_spatial, np.ndarray)
        self.assertEqual(similar_spatial.shape[1], 4)

        # Check that different weights lead to different selections
        # Spatial weight should prefer the trajectory with similar spatial pattern (westward)
        self.assertTrue(np.allclose(similar_spatial[:, 2], traj_slow_west[:, 2]))

        # Speed weight should prefer the trajectory with medium speed
        self.assertTrue(np.allclose(similar_speed[:, 2], traj_med_east[:, 2]))

        # Course weight should prefer the trajectory with same direction
        self.assertTrue(np.allclose(similar_course[:, 3], traj_slow_west[:, 3]))

    def test_prediction(self):
        """Test if model can predict the next points in a trajectory."""
        n_steps = 3
        input_length = 10
        input_traj = self.traj1[:input_length]
        predictions, similar_traj = self.model.predict(input_traj, n_steps)

        # Check shapes
        self.assertEqual(predictions.shape, (n_steps, 2))
        self.assertEqual(similar_traj.shape[1], 4)

        # Expected values are the next n_steps points from input trajectory
        expected = self.traj1[input_length : input_length + n_steps]

        # Predictions should be close to expected values (only lat/lon)
        self.assertTrue(
            np.allclose(predictions, expected[:, :2], atol=0.5),
            f"Predictions:\n{predictions}\nExpected:\n{expected[:, :2]}",
        )

    def test_prediction_with_short_trajectory(self):
        """Test prediction handling when similar trajectory is too short."""
        # Create a longer prediction request than available trajectory
        n_steps = 10
        predictions, similar_traj = self.model.predict(self.traj1[:3], n_steps)

        # Should still return arrays of correct type
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(similar_traj, np.ndarray)

        # Should have attempted to generate predictions
        self.assertEqual(predictions.shape[1], 2)

    def test_model_device_handling(self):
        """Test model device handling."""
        # Test CPU device
        model_cpu = TSSPL(self.trajectories, device="cpu")
        self.assertEqual(next(model_cpu.lstm_model.parameters()).device.type, "cpu")

        # Test CUDA device if available
        if torch.cuda.is_available():
            model_cuda = TSSPL(self.trajectories, device="cuda")
            self.assertEqual(next(model_cuda.lstm_model.parameters()).device.type, "cuda")

    def test_invalid_weights(self):
        """Test handling of invalid weights."""
        # Weights not summing to 1
        with self.assertRaises(ValueError):
            TSSPL(self.trajectories, w_d=0.5, w_s=0.5, w_c=0.5)

        # Negative weights
        with self.assertRaises(ValueError):
            TSSPL(self.trajectories, w_d=-0.1, w_s=0.5, w_c=0.6)

    def test_model_training(self):
        """Test model training process."""
        # Access the dataloader to get a batch
        dataloader = DataLoader(self.model.dataset, batch_size=2, shuffle=True)
        batch_X, batch_y = next(iter(dataloader))

        # Test forward pass
        self.model.lstm_model.train()
        predictions = self.model.lstm_model(batch_X.to(self.model.device))

        # Check prediction shape
        self.assertEqual(predictions.shape, (batch_X.shape[0], 2))

        # Test backward pass
        criterion = torch.nn.MSELoss()
        loss = criterion(predictions, batch_y.to(self.model.device))
        loss.backward()

        # Check that gradients were computed
        for param in self.model.lstm_model.parameters():
            self.assertIsNotNone(param.grad)

    def test_prediction_scaling(self):
        """Test that predictions are properly scaled."""
        n_steps = 3
        predictions, _ = self.model.predict(self.traj1, n_steps)

        # Get the range of the original trajectories
        all_lats = np.concatenate([traj[:, 0] for traj in self.trajectories])
        all_lons = np.concatenate([traj[:, 1] for traj in self.trajectories])

        lat_range = all_lats.max() - all_lats.min()
        lon_range = all_lons.max() - all_lons.min()

        # Predictions should be in similar range to original data
        pred_lat_range = predictions[:, 0].max() - predictions[:, 0].min()
        pred_lon_range = predictions[:, 1].max() - predictions[:, 1].min()

        self.assertLess(pred_lat_range, lat_range * 2)  # Allow some extrapolation
        self.assertLess(pred_lon_range, lon_range * 2)
