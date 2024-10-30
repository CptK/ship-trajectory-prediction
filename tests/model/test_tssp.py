from unittest import TestCase
from unittest.mock import patch

import numpy as np

from prediction.model.tssp import TSSP


class TestTSSP(TestCase):
    def setUp(self):
        # Create sample trajectories
        self.target = np.array(
            [
                [48.0, -125.0, 10.0, 180.0],  # lat, lon, sog, cog
                [48.1, -125.0, 10.0, 180.0],
                [48.2, -125.0, 10.0, 180.0],
            ]
        )

        self.similar = np.array(
            [
                [48.0, -125.1, 10.0, 180.0],
                [48.1, -125.1, 10.0, 180.0],
                [48.2, -125.1, 10.0, 180.0],
                [48.3, -125.1, 10.0, 180.0],
                [48.4, -125.1, 10.0, 180.0],
            ]
        )

        self.database = [self.similar]
        self.tssp = TSSP(self.database)

    def test_prediction_shape(self):
        """Test if prediction output has correct shape."""
        predictions, _ = self.tssp.predict(self.target, n_steps=2)
        self.assertEqual(predictions.shape, (2, 2))

    def test_constant_spatial_difference(self):
        """Test if spatial difference remains constant in predictions."""
        predictions, similar = self.tssp.predict(self.target, n_steps=2)

        # Calculate initial differences
        initial_lat_diff = self.target[-1, 0] - similar[len(self.target) - 1, 0]
        initial_lon_diff = self.target[-1, 1] - similar[len(self.target) - 1, 1]

        # Check if differences are maintained in predictions
        for i in range(len(predictions)):
            pred_idx = len(self.target) + i
            lat_diff = predictions[i, 0] - similar[pred_idx, 0]
            lon_diff = predictions[i, 1] - similar[pred_idx, 1]

            self.assertAlmostEqual(lat_diff, initial_lat_diff)
            self.assertAlmostEqual(lon_diff, initial_lon_diff)

    def test_insufficient_similar_trajectory(self):
        """Test behavior when similar trajectory is too short."""
        with patch("builtins.print") as mock_print:
            predictions, _ = self.tssp.predict(self.target, n_steps=10)
            mock_print.assert_called_once()
            self.assertEqual(len(predictions), 2)  # Only 2 points available after target length

    def test_weights_sum_to_one(self):
        """Test if weights sum to 1."""
        self.assertAlmostEqual(self.tssp.w_d + self.tssp.w_s + self.tssp.w_c, 1.0)
