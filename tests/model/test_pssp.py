from unittest import TestCase

import numpy as np

from prediction.model.pssp import PSSP
from prediction.preprocessing.utils import haversine


class TestPSSP(TestCase):
    def setUp(self):
        """Create test data that will be used by multiple tests."""
        # Create simple straight-line trajectories
        self.traj1 = np.array(
            [
                [48.0, -125.0, 10.0, 45.0],  # lat, lon, sog, cog
                [48.1, -124.9, 10.0, 45.0],
                [48.2, -124.8, 10.0, 45.0],
                [48.3, -124.7, 10.0, 45.0],
            ]
        )

        self.traj2 = np.array(
            [
                [48.1, -125.1, 10.0, 45.0],
                [48.2, -125.0, 10.0, 45.0],
                [48.3, -124.9, 10.0, 45.0],
                [48.4, -124.8, 10.0, 45.0],
            ]
        )

        self.database = [self.traj1, self.traj2]
        self.pssp = PSSP(self.database)

    def test_initialization(self):
        """Test initialization with various parameters."""
        # Test default initialization
        pssp = PSSP(self.database)
        self.assertEqual(pssp.w_d, 0.6)
        self.assertEqual(pssp.w_s, 0.2)
        self.assertEqual(pssp.w_c, 0.2)
        self.assertEqual(pssp.azimuth_threshold, 22.5)

        # Test custom weights
        pssp = PSSP(self.database, w_d=0.5, w_s=0.3, w_c=0.2)
        self.assertEqual(pssp.w_d, 0.5)
        self.assertEqual(pssp.w_s, 0.3)
        self.assertEqual(pssp.w_c, 0.2)

        # Test empty database
        with self.assertRaises(ValueError):
            PSSP([])

    def test_find_most_similar_point(self):
        """Test the similar point finding functionality."""
        target_point = np.array([48.05, -125.05, 10.0, 45.0])

        similar_point, traj_idx, point_idx = self.pssp._find_most_similar_point(target_point)

        # Check that we got a result
        self.assertIsNotNone(similar_point)
        self.assertIsNotNone(traj_idx)
        self.assertIsNotNone(point_idx)

        # Check that the found point is actually close to target
        distance = haversine(
            target_point[0], target_point[1], similar_point[0], similar_point[1]  # type: ignore
        )
        self.assertLess(distance, 10.0)  # Should be within 10km

        # Check that speed and course match
        self.assertAlmostEqual(target_point[2], similar_point[2])  # type: ignore # Speed
        self.assertAlmostEqual(target_point[3], similar_point[3])  # type: ignore # Course

    def test_get_next_point(self):
        """Test retrieving next point from trajectory."""
        # Test valid case
        next_point = self.pssp._get_next_point(0, 0)
        self.assertIsNotNone(next_point)
        np.testing.assert_array_equal(next_point, self.traj1[1])  # type: ignore

        # Test edge cases
        self.assertIsNone(self.pssp._get_next_point(None, 0))
        self.assertIsNone(self.pssp._get_next_point(0, None))
        self.assertIsNone(self.pssp._get_next_point(0, len(self.traj1) - 1))

    def test_predict(self):
        """Test the complete prediction process."""
        target_point = np.array([48.05, -125.05, 10.0, 45.0])
        n_steps = 3

        predictions, similar_points = self.pssp.predict(target_point, n_steps)

        # Check shapes
        self.assertEqual(predictions.shape, (n_steps, 2))
        self.assertEqual(similar_points.shape, (n_steps, self.database[0].shape[1]))

        # Check that predictions follow a reasonable pattern
        for i in range(1, len(predictions)):
            # Distance between consecutive predictions should be similar to
            # distance between consecutive points in original trajectories
            dist = haversine(
                predictions[i - 1, 0], predictions[i - 1, 1], predictions[i, 0], predictions[i, 1]
            )
            orig_dist = haversine(self.traj1[0, 0], self.traj1[0, 1], self.traj1[1, 0], self.traj1[1, 1])
            self.assertLess(abs(dist - orig_dist), 5.0)  # Within 5km tolerance

    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test invalid target point
        invalid_target = np.array([48.0, -125.0])  # Missing sog and cog
        with self.assertRaises(ValueError):
            self.pssp.predict(invalid_target, 3)

        # Test invalid n_steps
        valid_target = np.array([48.0, -125.0, 10.0, 45.0])
        predictions, _ = self.pssp.predict(valid_target, 0)
        self.assertEqual(len(predictions), 0)

    def test_edge_cases(self):
        """Test edge cases and special scenarios."""
        # Test when target is exactly on a trajectory point
        target_point = self.traj1[0].copy()
        predictions, _ = self.pssp.predict(target_point, 2)
        self.assertEqual(len(predictions), 2)

        # Test prediction with single-point trajectories
        pssp = PSSP([np.array([[48.0, -125.0, 10.0, 45.0]])])
        target_point = np.array([48.0, -125.0, 10.0, 45.0])
        predictions, similar_points = pssp.predict(target_point, 2)
        self.assertEqual(len(predictions), 0)  # Should fail to predict due to short trajectory
