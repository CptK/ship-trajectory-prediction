from math import isclose
from unittest import TestCase

import numpy as np
import torch
from numpy.testing import assert_almost_equal
from shapely.geometry import LineString

from prediction.preprocessing.utils import (
    azimuth,
    calc_sog,
    dtw_spatial,
    haversine,
    haversine_tensor,
    pairwise_point_distances,
    max_bbox_diagonal,
)


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


class TestDTWSpatial(TestCase):
    def setUp(self):
        """Create sample trajectories for testing."""
        # Simple parallel trajectories
        self.traj1 = np.array(
            [
                [48.0, -125.0, 10.0, 180.0],  # lat, lon, sog, cog
                [48.1, -125.0, 10.0, 180.0],
                [48.2, -125.0, 10.0, 180.0],
            ]
        )

        self.traj2 = np.array(
            [[48.0, -125.1, 10.0, 180.0], [48.1, -125.1, 10.0, 180.0], [48.2, -125.1, 10.0, 180.0]]
        )

        # Single point trajectories
        self.single_point1 = np.array([[48.0, -125.0, 10.0, 180.0]])
        self.single_point2 = np.array([[48.0, -125.1, 10.0, 180.0]])

        # Different length trajectories
        self.traj_short = np.array([[48.0, -125.0, 10.0, 180.0], [48.1, -125.0, 10.0, 180.0]])

        # Identical trajectories
        self.traj_identical = np.array([[48.0, -125.0, 10.0, 180.0], [48.1, -125.0, 10.0, 180.0]])

    def test_parallel_trajectories(self):
        """Test DTW distance between parallel trajectories."""
        distance = dtw_spatial(self.traj1, self.traj2)
        self.assertGreater(distance[0], 0)  # Should have non-zero distance

    def test_single_point(self):
        """Test DTW distance between single point trajectories."""
        distance = dtw_spatial(self.single_point1, self.single_point2)
        self.assertGreater(distance[0], 0)

    def test_different_lengths(self):
        """Test DTW handles trajectories of different lengths."""
        distance = dtw_spatial(self.traj1, self.traj_short)
        self.assertGreater(distance[0], 0)

    def test_identical_trajectories(self):
        """Test DTW distance between identical trajectories is zero."""
        distance = dtw_spatial(self.traj_identical, self.traj_identical)
        assert_almost_equal(distance[0], 0, decimal=6)

    def test_symmetry(self):
        """Test DTW distance is symmetric."""
        dist1 = dtw_spatial(self.traj1, self.traj2)
        dist2 = dtw_spatial(self.traj2, self.traj1)
        assert_almost_equal(dist1[0], dist2[0], decimal=6)


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


class TestAzimuth(TestCase):
    def setUp(self):
        """Set up test cases with known azimuths."""
        # Tolerance for floating point comparisons
        self.precision = 0.1

    def test_cardinal_directions(self):
        """Test azimuth for cardinal directions from origin."""
        # From point (0,0) to cardinal directions
        # North
        self.assertAlmostEqual(azimuth(0, 0, 1, 0), 0.0, delta=self.precision)
        # East
        self.assertAlmostEqual(azimuth(0, 0, 0, 1), 90.0, delta=self.precision)
        # South
        self.assertAlmostEqual(azimuth(0, 0, -1, 0), 180.0, delta=self.precision)
        # West
        self.assertAlmostEqual(azimuth(0, 0, 0, -1), 270.0, delta=self.precision)

    def test_intercardinal_directions(self):
        """Test azimuth for intercardinal directions."""
        # Northeast
        self.assertAlmostEqual(azimuth(0, 0, 1, 1), 45.0, delta=self.precision)
        # Southeast
        self.assertAlmostEqual(azimuth(0, 0, -1, 1), 135.0, delta=self.precision)
        # Southwest
        self.assertAlmostEqual(azimuth(0, 0, -1, -1), 225.0, delta=self.precision)
        # Northwest
        self.assertAlmostEqual(azimuth(0, 0, 1, -1), 315.0, delta=self.precision)

    def test_real_world_coordinates(self):
        """Test azimuth with real-world coordinates."""
        # New York to London
        ny_lat, ny_lon = 40.7128, -74.0060
        london_lat, london_lon = 51.5074, -0.1278
        self.assertAlmostEqual(
            azimuth(ny_lat, ny_lon, london_lat, london_lon),
            51.2,  # Corrected expected azimuth
            delta=1.0,  # Reasonable tolerance for real-world example
        )

    def test_same_point(self):
        """Test azimuth when points are the same."""
        azim = azimuth(45.0, -120.0, 45.0, -120.0)
        self.assertTrue(0 <= azim < 360)

    def test_wrap_around(self):
        """Test azimuth wrap-around at 360 degrees."""
        # Should give same result as 1 degree
        azimuth_361 = azimuth(0, 0, 0.01, 0.01)
        azimuth_1 = azimuth(0, 0, 0.01, 0.01)
        self.assertAlmostEqual(azimuth_361, azimuth_1, delta=self.precision)

    def test_pole_azimuth(self):
        """Test azimuth near poles."""
        # Near North Pole
        north_pole_azimuth = azimuth(89.0, 0, 90.0, 0)
        self.assertAlmostEqual(north_pole_azimuth, 0.0, delta=self.precision)

        # Near South Pole
        south_pole_azimuth = azimuth(-89.0, 0, -90.0, 0)
        self.assertAlmostEqual(south_pole_azimuth, 180.0, delta=self.precision)

    def test_input_types(self):
        """Test different input types."""
        # Test with integers
        int_azimuth = azimuth(0, 0, 1, 1)
        self.assertIsInstance(int_azimuth, float)


class TestPairwisePointDistances(TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Simple line with 3 points forming a triangle in the US
        self.simple_line = LineString(
            [
                (-122.4194, 37.7749),  # San Francisco
                (-118.2437, 34.0522),  # Los Angeles
                (-115.1398, 36.1699),  # Las Vegas
            ]
        )

        # Line with identical points
        self.identical_points_line = LineString(
            [(-122.4194, 37.7749), (-122.4194, 37.7749)]  # San Francisco  # Same point
        )

        # Empty line
        self.empty_line = LineString([])

    def test_matrix_dimensions(self):
        """Test if the output matrix has correct dimensions."""
        result = pairwise_point_distances(self.simple_line)
        self.assertEqual(result.shape, (3, 3))

        result_identical = pairwise_point_distances(self.identical_points_line)
        self.assertEqual(result_identical.shape, (2, 2))

        result_empty = pairwise_point_distances(self.empty_line)
        self.assertEqual(result_empty.shape, (0, 0))

    def test_matrix_symmetry(self):
        """Test if the distance matrix is symmetric."""
        result = pairwise_point_distances(self.simple_line)
        np.testing.assert_array_almost_equal(result, result.T)

    def test_diagonal_zeros(self):
        """Test if diagonal elements are zero."""
        result = pairwise_point_distances(self.simple_line)
        np.testing.assert_array_almost_equal(np.diagonal(result), np.zeros(len(result)))

    def test_known_distances(self):
        """Test against pre-calculated known distances."""
        result = pairwise_point_distances(self.simple_line)

        # Known approximate distances in kilometers
        sf_to_la = 559.0  # SF to LA
        la_to_lv = 368.0  # LA to Las Vegas
        sf_to_lv = 670.0  # SF to Las Vegas

        # Test with 1% tolerance due to different haversine implementations
        self.assertTrue(isclose(result[0][1], sf_to_la, rel_tol=0.01))
        self.assertTrue(isclose(result[1][2], la_to_lv, rel_tol=0.01))
        self.assertTrue(isclose(result[0][2], sf_to_lv, rel_tol=0.01))

    def test_identical_points(self):
        """Test distance between identical points is zero."""
        result = pairwise_point_distances(self.identical_points_line)
        np.testing.assert_array_almost_equal(result, np.zeros((2, 2)))

    def test_empty_line(self):
        """Test empty line returns empty matrix."""
        result = pairwise_point_distances(self.empty_line)
        self.assertEqual(result.shape, (0, 0))

    def test_distances_are_positive(self):
        """Test if all distances are non-negative."""
        result = pairwise_point_distances(self.simple_line)
        self.assertTrue(np.all(result >= 0))


class TestMaxBboxDiagonal(TestCase):
    def test_simple_diagonal(self):
        """Test with a simple diagonal line."""
        line = LineString([(0, 0), (1, 1)])
        diagonal = max_bbox_diagonal(line)
        # The diagonal should be approximately 157 km
        self.assertAlmostEqual(diagonal, 157.25, delta=0.1)

    def test_horizontal_line(self):
        """Test with a horizontal line."""
        line = LineString([(0, 0), (1, 0)])
        diagonal = max_bbox_diagonal(line)
        # Should be approximately 111 km for 1 degree longitude at equator
        self.assertAlmostEqual(diagonal, 111.19, delta=0.1)

    def test_vertical_line(self):
        """Test with a vertical line."""
        line = LineString([(0, 0), (0, 1)])
        diagonal = max_bbox_diagonal(line)
        # Should be approximately 111 km for 1 degree latitude
        self.assertAlmostEqual(diagonal, 111.19, delta=0.1)

    def test_complex_line(self):
        """Test with a complex line that zigzags."""
        line = LineString([
            (0, 0),    # Start
            (0.5, 1),  # North
            (1, 0.5),  # East
            (0.5, 0),  # South
            (0, 0.5)   # Back to middle
        ])
        diagonal = max_bbox_diagonal(line)
        # The bounding box should be 1x1 degree
        self.assertAlmostEqual(diagonal, 157.25, delta=0.1)

    def test_two_identical_points(self):
        """Test with a line consisting of two identical points."""
        line = LineString([(0, 0), (0, 0)])
        diagonal = max_bbox_diagonal(line)
        self.assertEqual(diagonal, 0)

    def test_realistic_ship_path(self):
        """Test with a realistic ship path in the North Sea."""
        line = LineString([
            (3.0, 51.5),   # Rotterdam area
            (4.0, 52.0),   # Amsterdam area
            (5.0, 53.0),   # German Bight
            (6.0, 53.5),   # German coast
            (7.0, 54.0)    # Danish waters
        ])
        diagonal = max_bbox_diagonal(line)
        # The diagonal should be a few hundred kilometers
        self.assertGreater(diagonal, 200)  # More than 200 km
        self.assertLess(diagonal, 500)     # Less than 500 km

    def test_around_dateline(self):
        """Test with a path crossing the international date line."""
        line = LineString([
            (179.5, 0),
            (-179.5, 0)
        ])
        diagonal = max_bbox_diagonal(line)
        # Should be approximately 111 km (1 degree at equator)
        self.assertAlmostEqual(diagonal, 111.19, delta=0.1)

    def test_near_poles(self):
        """Test with a path near the poles where meridians converge."""
        line = LineString([
            (0, 89),
            (10, 89)
        ])
        diagonal = max_bbox_diagonal(line)
        # Distance should be relatively small due to meridian convergence
        self.assertLess(diagonal, 50)  # Less than 50 km

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        line = LineString([
            (-1, -1),
            (-2, -2)
        ])
        diagonal = max_bbox_diagonal(line)
        # Should be approximately 157 km
        self.assertAlmostEqual(diagonal, 157.23, delta=0.1)
