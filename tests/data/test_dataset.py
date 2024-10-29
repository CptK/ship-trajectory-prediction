from collections import defaultdict
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from prediction.data.dataset import MaskedAISDataset


class TestMaskedAISDataset(TestCase):
    """Test suite for the MaskedAISDataset class."""

    def setUp(self):
        """Set up test data."""
        # Create sample trajectory
        points = [(i, i) for i in range(50)]  # 50 points trajectory
        line = LineString(points)

        # Create timestamps
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(minutes=i * 5) for i in range(50)]

        # Create test dataframe
        self.df = pd.DataFrame(
            {
                "geometry": [line],
                "velocities": [[10.0] * 50],
                "orientations": [[45.0] * 50],
                "timestamps": [timestamps],
            }
        )

        # Create dataset instance with different ratios
        self.dataset = MaskedAISDataset(
            df=self.df,
            max_seq_len=100,
            random_mask_ratio=0.15,
            consecutive_mask_ratio=0.8,
            end_mask_ratio=0.5,
            mask_token=-1.0,
            min_masks=1,
            pad_token=-999.0,
            random_mask_prob=0.2,
            consecutive_mask_prob=0.3,
            end_mask_prob=0.5,
            normalize=True,
            include_speed=True,
            include_orientation=True,
            include_timestamps=True,
        )

    def test_initialization(self):
        """Test dataset initialization and parameter validation."""
        # Test valid initialization
        self.assertEqual(self.dataset.random_mask_ratio, 0.15)
        self.assertEqual(self.dataset.consecutive_mask_ratio, 0.8)
        self.assertEqual(self.dataset.end_mask_ratio, 0.5)

        # Test probability validation
        with self.assertRaises(ValueError):
            MaskedAISDataset(
                df=self.df,
                max_seq_len=100,
                random_mask_prob=0.2,
                consecutive_mask_prob=0.2,
                end_mask_prob=0.2,  # Sum < 1
            )

        with self.assertRaises(ValueError):
            MaskedAISDataset(
                df=self.df,
                max_seq_len=100,
                random_mask_prob=-0.1,  # Negative probability
                consecutive_mask_prob=0.5,
                end_mask_prob=0.6,
            )

    def test_random_masking(self):
        """Test random masking strategy."""
        seq_len = 50
        mask = self.dataset._create_random_masks(seq_len)

        # Check number of masked positions
        expected_masks = max(self.dataset.min_masks, int(seq_len * self.dataset.random_mask_ratio))
        self.assertEqual(mask.sum(), expected_masks)

        # Check mask distribution
        self.assertEqual(len(mask), seq_len)
        self.assertEqual(mask.dtype, bool)

    def test_consecutive_masking(self):
        """Test consecutive masking strategy."""
        seq_len = 50
        mask = self.dataset._create_consecutive_masks(seq_len)

        # Check number of masked positions
        expected_masks = max(self.dataset.min_masks, int(seq_len * self.dataset.consecutive_mask_ratio))
        self.assertEqual(mask.sum(), expected_masks)

        # Verify consecutiveness
        masked_positions = np.where(mask)[0]
        self.assertEqual(
            len(masked_positions),
            masked_positions[-1] - masked_positions[0] + 1,
            "Masked positions should be consecutive",
        )

    def test_end_masking(self):
        """Test end masking strategy."""
        seq_len = 50
        mask = self.dataset._create_end_masks(seq_len)

        # Check number of masked positions
        expected_masks = max(self.dataset.min_masks, int(seq_len * self.dataset.end_mask_ratio))
        self.assertEqual(mask.sum(), expected_masks)

        # Verify end positions are masked
        self.assertTrue(np.all(mask[-expected_masks:]), "End positions should be masked")
        self.assertTrue(np.all(~mask[:-expected_masks]), "Non-end positions should not be masked")

    def test_strategy_distribution(self):
        """Test that strategies are selected according to their probabilities."""
        n_samples = 1000
        strategy_counts = defaultdict(int)  # type: ignore

        # Monkey patch _create_masks to count strategy usage
        original_create_masks = self.dataset._create_masks

        def counting_create_masks(seq_len):
            strategy_idx = np.random.choice(3, p=self.dataset.strategy_probs)
            strategy_counts[strategy_idx] += 1
            return original_create_masks(seq_len)

        self.dataset._create_masks = counting_create_masks  # type: ignore

        # Generate samples
        for _ in range(n_samples):
            self.dataset[0]

        self.assertAlmostEqual(
            strategy_counts[0] / n_samples, 0.2, delta=0.05, msg="Random masking frequency should be ~20%"
        )
        self.assertAlmostEqual(
            strategy_counts[1] / n_samples,
            0.3,
            delta=0.05,
            msg="Consecutive masking frequency should be ~30%",
        )
        self.assertAlmostEqual(
            strategy_counts[2] / n_samples, 0.5, delta=0.05, msg="End masking frequency should be ~50%"
        )

        # Restore original method
        self.dataset._create_masks = original_create_masks  # type: ignore

    def test_pad_sequence(self):
        """Test sequence padding and truncation."""
        # Test padding (sequence shorter than max_seq_len)
        short_seq = np.random.randn(30, 6)  # 30 timesteps, 6 features
        padded = self.dataset._pad_sequence(short_seq)

        self.assertEqual(padded.shape, (self.dataset.max_seq_len, 6))
        self.assertTrue(np.all(padded[:30] == short_seq))
        self.assertTrue(np.all(padded[30:] == self.dataset.pad_token))

        # Test truncation (sequence longer than max_seq_len)
        long_seq = np.random.randn(150, 6)  # 150 timesteps, 6 features
        truncated = self.dataset._pad_sequence(long_seq)

        self.assertEqual(truncated.shape, (self.dataset.max_seq_len, 6))
        self.assertTrue(np.all(truncated == long_seq[: self.dataset.max_seq_len]))

        # Test exact length (no padding or truncation needed)
        exact_seq = np.random.randn(self.dataset.max_seq_len, 6)
        result = self.dataset._pad_sequence(exact_seq)

        self.assertEqual(result.shape, (self.dataset.max_seq_len, 6))
        self.assertTrue(np.all(result == exact_seq))

    def test_getitem_output(self):
        """Test the complete output of __getitem__."""
        masked_input, attention_mask, target = self.dataset[0]

        # Check shapes
        self.assertEqual(
            masked_input.shape, (self.dataset.max_seq_len, 6)  # 2(lat,lon) + 1(speed) + 2(sin,cos) + 1(time)
        )
        self.assertEqual(attention_mask.shape, (self.dataset.max_seq_len,))
        self.assertEqual(target.shape, (self.dataset.max_seq_len, 6))

        # Check types
        self.assertEqual(masked_input.dtype, np.float32)
        self.assertEqual(attention_mask.dtype, bool)
        self.assertEqual(target.dtype, np.float32)

        # Check padding
        self.assertTrue(np.all(attention_mask[:50]))  # Real sequence positions
        self.assertTrue(np.all(~attention_mask[50:]))  # Padding positions
        self.assertTrue(np.all(masked_input[50:] == self.dataset.pad_token))
        self.assertTrue(np.all(target[50:] == self.dataset.pad_token))

        # Verify mask tokens are only in non-padded regions
        mask_positions = (masked_input == self.dataset.mask_token).any(axis=1)
        self.assertTrue(np.all(mask_positions[mask_positions] < 50))

    def test_getitem_sequence_length(self):
        """Test __getitem__ handling of different sequence lengths."""
        # Create a longer trajectory
        long_points = [(i, i) for i in range(150)]  # 150 points > max_seq_len
        line = LineString(long_points)

        # Create corresponding timestamps
        base_time = datetime(2024, 1, 1)
        timestamps = [base_time + timedelta(minutes=i * 5) for i in range(150)]

        # Create test dataframe with long sequence
        df_long = pd.DataFrame(
            {
                "geometry": [line],
                "velocities": [[10.0] * 150],
                "orientations": [[45.0] * 150],
                "timestamps": [timestamps],
            }
        )

        # Create dataset with the long sequence
        dataset_long = MaskedAISDataset(
            df=df_long,
            max_seq_len=100,  # Same as original
            random_mask_ratio=0.15,
            consecutive_mask_ratio=0.8,
            end_mask_ratio=0.5,
            mask_token=-1.0,
            min_masks=1,
            pad_token=-999.0,
        )

        # Get a sample
        masked_input, attention_mask, target = dataset_long[0]

        # Verify shapes
        self.assertEqual(masked_input.shape, (100, 6))
        self.assertEqual(attention_mask.shape, (100,))
        self.assertEqual(target.shape, (100, 6))

        # Verify we kept the last 100 points
        # We can check this by verifying the positions in the target sequence
        # The positions should be the last 100 points from the original 150
        original_positions = np.array(long_points)[-100:, :]  # Last 100 points
        target_positions = target[:, :2]  # Just the lat/lon columns

        # Need to account for normalization if it's enabled
        if dataset_long.normalize:
            original_positions = original_positions / np.array([90, 180])

        # Verify positions match (up to floating point precision)
        np.testing.assert_allclose(
            target_positions[attention_mask],  # Only compare non-padded positions
            original_positions,
            rtol=1e-5,
            err_msg="Truncated sequence should contain the last max_seq_len points",
        )

        # Verify attention mask
        self.assertTrue(np.all(attention_mask))  # All positions should be real data

        # Test exact length sequence
        exact_points = [(i, i) for i in range(100)]  # Exactly max_seq_len points
        line_exact = LineString(exact_points)
        timestamps_exact = [base_time + timedelta(minutes=i * 5) for i in range(100)]

        df_exact = pd.DataFrame(
            {
                "geometry": [line_exact],
                "velocities": [[10.0] * 100],
                "orientations": [[45.0] * 100],
                "timestamps": [timestamps_exact],
            }
        )

        dataset_exact = MaskedAISDataset(
            df=df_exact,
            max_seq_len=100,
            random_mask_ratio=0.15,
            consecutive_mask_ratio=0.8,
            end_mask_ratio=0.5,
            mask_token=-1.0,
            min_masks=1,
            pad_token=-999.0,
        )

        # Get a sample
        masked_input_exact, attention_mask_exact, target_exact = dataset_exact[0]

        # Verify no truncation occurred
        self.assertEqual(masked_input_exact.shape, (100, 6))
        self.assertTrue(np.all(attention_mask_exact))
