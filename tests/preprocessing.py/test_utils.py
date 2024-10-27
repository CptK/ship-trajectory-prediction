from unittest import TestCase

import torch
import numpy as np

from prediction.preprocessing.utils import haversine_tensor, haversine, calc_sog


class TestHaversineTensor(TestCase):
    def setUp(self):
        red_opt = ["sum", "mean", "none"]

        self.reduction_options = [(traj_red, tensor_red) for traj_red in red_opt for tensor_red in red_opt]

    def _get_expected_shape(self, batch_size, seq_len, traj_red, tensor_red):
        if traj_red == "none":
            if tensor_red == "none":
                return (batch_size, seq_len)
            else:
                return (seq_len,)
        else:  # traj_red is 'sum' or 'mean'
            if tensor_red == "none":
                return (batch_size,)
            else:  # tensor_red is 'sum' or 'mean'
                return ()

    def test_single_trajectory_single_point(self):
        # create trajectory of shape (1, 1, 2)
        traj1 = torch.tensor([[[0, 0]]], dtype=torch.float32)
        traj2 = torch.tensor([[[0, 0]]], dtype=torch.float32)

        for traj_red, tensor_red in self.reduction_options:
            dist = haversine_tensor(traj1, traj2, traj_reduction=traj_red, tensor_reduction=tensor_red)
            self.assertEqual(dist.shape, self._get_expected_shape(1, 1, traj_red, tensor_red))
            self.assertEqual(dist.item(), 0)

    def test_single_trajectory_multiple_points(self):
        # create trajectory of shape (1, 3, 2)
        traj1 = torch.tensor([[[0, 0], [0, 1], [1, 1]]], dtype=torch.float32)
        traj2 = torch.tensor([[[0, 0], [0, 1], [1, 1]]], dtype=torch.float32)

        for traj_red, tensor_red in self.reduction_options:
            expected_shape = self._get_expected_shape(1, 3, traj_red, tensor_red)
            expected_dist = torch.zeros(expected_shape)

            dist = haversine_tensor(traj1, traj2, traj_reduction=traj_red, tensor_reduction=tensor_red)
            self.assertTrue(torch.allclose(dist, expected_dist))

    def test_multiple_trajectories_multiple_points(self):
        # create trajectories of shape (2, 3, 2)
        traj1 = torch.tensor([[[0, 0], [0, 1], [1, 1]], [[2, 2], [3, 3], [4, 4]]], dtype=torch.float32)
        traj2 = torch.tensor([[[0, 0], [0, 1], [1, 1]], [[2, 2], [3, 3], [4, 4]]], dtype=torch.float32)

        for traj_red, tensor_red in self.reduction_options:
            expected_shape = self._get_expected_shape(2, 3, traj_red, tensor_red)
            expected_dist = torch.zeros(expected_shape)

            dist = haversine_tensor(traj1, traj2, traj_reduction=traj_red, tensor_reduction=tensor_red)
            self.assertTrue(torch.allclose(dist, expected_dist))


class TestHaversine(TestCase):
    def test_0_distance(self):
        """Test that identical points return 0 distance"""
        dist = haversine(0, 0, 0, 0)
        self.assertEqual(dist, 0)

        # Test at non-zero coordinates
        dist = haversine(45.0, 45.0, 45.0, 45.0)
        self.assertEqual(dist, 0)

    def test_short_distances(self):
        """Test accurate calculation of short distances (~100m)"""
        # Near equator
        dist = haversine(0, 0, 0, 0.0009)
        self.assertAlmostEqual(dist, 0.1, delta=0.01)

        # At higher latitude (distance between meridians is shorter)
        dist = haversine(60, 0, 60, 0.0018)
        self.assertAlmostEqual(dist, 0.1, delta=0.01)

    def test_medium_distances(self):
        """Test accurate calculation of medium distances (~100km)"""
        dist = haversine(0, 0, 0.9, 0)
        self.assertAlmostEqual(dist, 100, delta=1)

    def test_long_distances(self):
        """Test accurate calculation of long distances"""
        # New York to London (approximate coordinates)
        dist = haversine(40.7128, -74.0060, 51.5074, -0.1278)
        self.assertAlmostEqual(dist, 5570, delta=50)  # ~5570 km

        # antipodal points (maximum possible distance)
        dist = haversine(0, 0, 0, 180)
        self.assertAlmostEqual(dist, 20015, delta=50)  # half Earth's circumference

    def test_crossing_poles(self):
        """Test calculations across poles"""
        # North pole to point on equator
        dist = haversine(90, 0, 0, 0)
        self.assertAlmostEqual(dist, 10007.5, delta=50)  # quarter Earth's circumference

    def test_crossing_dateline(self):
        """Test calculations across the international date line"""
        # Points on either side of the international date line
        dist = haversine(0, 179, 0, -179)
        self.assertAlmostEqual(dist, 222.4, delta=10)  # Should be ~222.4 km

    def test_input_validation(self):
        """Test that invalid coordinates raise appropriate errors"""
        # Latitude outside [-90, 90]
        with self.assertRaises(ValueError):
            haversine(91, 0, 0, 0)

        # Longitude outside [-180, 180]
        with self.assertRaises(ValueError):
            haversine(0, 181, 0, 0)

    def test_coordinate_types(self):
        """Test handling of different numeric types"""
        # Test with integers
        dist_int = haversine(0, 0, 1, 1)
        # Test with floats
        dist_float = haversine(0.0, 0.0, 1.0, 1.0)
        # Should give same results
        self.assertAlmostEqual(dist_int, dist_float, delta=0.0001)

    def test_numerical_stability(self):
        """Test numerical stability for very small distances"""
        # Very small distance
        dist = haversine(0, 0, 0, 0.000001)
        self.assertGreaterEqual(dist, 0)

        # Almost antipodal points
        dist = haversine(0, 0, 0, 179.9999999)
        # Earth's circumference is approximately 40,075 km
        # So half of it is 20,037.5 km
        self.assertLess(dist, 20037.5)  # Should be less than half Earth's circumference


class TestSpeedOverGround(TestCase):
    def test_stationary(self):
        """Test that identical points result in zero speed"""
        speed = calc_sog(0, 0, 0, 0, 3600)  # 1 hour
        self.assertAlmostEqual(speed, 0, places=6)

        # Test at non-zero coordinates
        speed = calc_sog(45.0, 45.0, 45.0, 45.0, 1800)  # 30 minutes
        self.assertAlmostEqual(speed, 0, places=6)

    def test_known_speeds(self):
        """Test speeds with known distances and times"""
        # Test 10 km/h speed (moving north)
        # In 1 hour (3600 seconds), should travel 10 km
        # ~0.09 degrees latitude = 10 km
        speed = calc_sog(0, 0, 0.09, 0, 3600)
        self.assertAlmostEqual(speed, 10, delta=0.1)  # 10 km/h with 0.1 km/h tolerance

        # Test approximately 20 knots (~37 km/h)
        # In 1 hour, should travel 37 km
        # ~0.333 degrees latitude = 37 km
        speed = calc_sog(0, 0, 0.333, 0, 3600)
        self.assertAlmostEqual(speed, 37, delta=0.5)  # 37 km/h with 0.5 km/h tolerance

    def test_diagonal_movement(self):
        """Test speeds when moving diagonally"""
        # Moving northeast at roughly 10 km/h
        speed = calc_sog(0, 0, 0.0635, 0.0635, 3600)
        self.assertAlmostEqual(speed, 10, delta=0.2)

    def test_time_variations(self):
        """Test same distance over different time periods"""
        # 10 km distance
        lat2 = 0.09  # approximately 10 km north

        # 1 hour = 10 km/h
        speed1 = calc_sog(0, 0, lat2, 0, 3600)
        # 30 minutes = 20 km/h
        speed2 = calc_sog(0, 0, lat2, 0, 1800)
        # 15 minutes = 40 km/h
        speed3 = calc_sog(0, 0, lat2, 0, 900)

        self.assertAlmostEqual(speed1, 10, delta=0.1)
        self.assertAlmostEqual(speed2, 20, delta=0.2)
        self.assertAlmostEqual(speed3, 40, delta=0.4)

    def test_zero_time(self):
        """Test handling of zero time difference"""
        speed = calc_sog(0, 0, 0.09, 0, 0)
        self.assertFalse(np.isinf(speed))  # Should not be infinite
        self.assertTrue(np.isfinite(speed))  # Should be finite

        # For very small time differences, speed should be very large but finite
        speed = calc_sog(0, 0, 0.09, 0, 1e-10)
        self.assertTrue(np.isfinite(speed))

    def test_negative_time(self):
        """Test that negative time differences raise an error"""
        with self.assertRaises(ValueError):
            calc_sog(0, 0, 0.09, 0, -3600)

    def test_high_speeds(self):
        """Test realistic high speeds"""
        # Test aircraft speed ~900 km/h
        # In 1 hour should travel ~900 km
        # ~8.1 degrees latitude = 900 km
        speed = calc_sog(0, 0, 8.1, 0, 3600)
        self.assertAlmostEqual(speed, 900, delta=10)

    def test_different_hemispheres(self):
        """Test speeds when crossing hemispheres"""
        # Moving from southern to northern hemisphere
        speed = calc_sog(-1, 0, 1, 0, 3600)  # 2 degrees in 1 hour
        self.assertAlmostEqual(speed, 222.4, delta=5)  # ~222.4 km/h

        # Moving across equator diagonally
        speed = calc_sog(-1, -1, 1, 1, 3600)
        self.assertAlmostEqual(speed, 314.3, delta=5)  # ~314.3 km/h (√2 times more)

    def test_input_validation(self):
        """Test that invalid inputs raise appropriate errors"""
        # Invalid latitude
        with self.assertRaises(ValueError):
            calc_sog(91, 0, 0, 0, 3600)

        # Invalid longitude
        with self.assertRaises(ValueError):
            calc_sog(0, 181, 0, 0, 3600)

        # Invalid time (should be non-negative)
        with self.assertRaises(ValueError):
            calc_sog(0, 0, 0, 0, -1)

    def test_precision(self):
        """Test handling of very small movements"""
        # Very small movement
        speed = calc_sog(0, 0, 0.000001, 0, 3600)
        self.assertGreater(speed, 0)
        self.assertTrue(np.isfinite(speed))
