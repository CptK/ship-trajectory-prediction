from unittest import TestCase

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from prediction.preprocessing.trajectory_resampling import (
    _calculate_dtw_distance,
    _interpolate_circular,
    _interpolate_point_on_line,
    _linestring_to_coords,
    _process_single_trajectory,
    compare_trajectory_pairs,
    resample_trajectories,
)


class TestTrajectoryResampling(TestCase):
    def setUp(self):
        """Create test data that will be used across multiple tests."""
        # Create a simple test trajectory
        self.x = [0.0, 1.0, 2.0]
        self.y = [0.0, 1.0, 2.0]
        self.times = [0, 180, 360]  # 0, 3, and 6 minutes
        self.velocities = [10.0, 12.0, 14.0]
        self.orientations = [0.0, 45.0, 90.0]
        self.statuses = [1, 1, 1]

        self.line = LineString(zip(self.x, self.y))
        self.df = pd.DataFrame(
            {
                "geometry": [self.line],
                "timestamps": [self.times],
                "velocities": [self.velocities],
                "orientations": [self.orientations],
                "statuses": [self.statuses],
            }
        )

    def test_interpolate_point_on_line(self):
        """Test point interpolation at various positions along a line."""
        line = LineString([(0, 0), (1, 1)])

        # Test start point
        point = _interpolate_point_on_line(line, 0.0)
        self.assertEqual((point.x, point.y), (0, 0))

        # Test end point
        point = _interpolate_point_on_line(line, line.length)
        self.assertEqual((point.x, point.y), (1, 1))

        # Test midpoint
        point = _interpolate_point_on_line(line, line.length / 2)
        self.assertAlmostEqual(point.x, 0.5)
        self.assertAlmostEqual(point.y, 0.5)

        # Test beyond line bounds
        point = _interpolate_point_on_line(line, -1.0)
        self.assertEqual((point.x, point.y), (0, 0))
        point = _interpolate_point_on_line(line, line.length + 1.0)
        self.assertEqual((point.x, point.y), (1, 1))

    def test_interpolate_circular(self):
        """Test circular interpolation for various angle combinations."""
        # Test simple interpolation
        result = _interpolate_circular(0, 90, 0.5)
        self.assertEqual(result, 45)

        # Test wrap around 360
        result = _interpolate_circular(350, 10, 0.5)
        self.assertEqual(result, 0)

        # Test negative angles
        result = _interpolate_circular(-90, 90, 0.5)
        self.assertEqual(result, 180)

        # Test fractions other than 0.5
        result = _interpolate_circular(0, 90, 0.25)
        self.assertEqual(result, 22.5)

        # Test full circle
        result = _interpolate_circular(0, 360, 0.5)
        self.assertEqual(result, 0)

    def test_process_single_trajectory(self):
        """Test processing of a single trajectory with various checks."""
        result = _process_single_trajectory(self.df.iloc[0], interval_seconds=180)  # 3-minute intervals

        # Check that we get a Series back
        self.assertIsInstance(result, pd.Series)

        # Check that all expected fields are present
        expected_fields = ["geometry", "timestamps", "velocities", "orientations", "statuses"]
        for field in expected_fields:
            self.assertIn(field, result.index)

        # Check the number of points
        self.assertEqual(len(result["timestamps"]), 3)

        # Check if times are at correct intervals
        self.assertEqual(result["timestamps"], [0, 180, 360])

        # Check geometry
        coords = list(result["geometry"].coords)
        self.assertAlmostEqual(coords[0][0], 0.0)
        self.assertAlmostEqual(coords[-1][0], 2.0)

    def test_resample_trajectories(self):
        """Test resampling of entire DataFrame."""
        resampled_df = resample_trajectories(self.df, interval_minutes=3)

        # Check DataFrame structure
        self.assertEqual(len(resampled_df), 1)
        self.assertTrue(isinstance(resampled_df, pd.DataFrame))

        # Check expected columns
        expected_columns = ["geometry", "timestamps", "velocities", "orientations", "statuses"]
        for col in expected_columns:
            self.assertIn(col, resampled_df.columns)

        # Check resampled data
        row = resampled_df.iloc[0]
        self.assertEqual(len(row["timestamps"]), 3)  # Should have 3 points for 0, 3, and 6 minutes
        self.assertEqual(len(row["velocities"]), 3)
        self.assertEqual(len(row["orientations"]), 3)
        self.assertEqual(len(row["statuses"]), 3)

    def test_edge_cases(self):
        """Test edge cases and potential error conditions."""
        # Test empty DataFrame
        empty_df = pd.DataFrame(columns=["geometry", "timestamps", "velocities", "orientations", "statuses"])
        result = resample_trajectories(empty_df)
        self.assertEqual(len(result), 0)


class TestTrajectoryComparison(TestCase):
    def setUp(self):
        """Set up test cases that will be used across multiple tests."""
        # Create simple test LineStrings
        self.line1 = LineString([(0, 0), (1, 1), (2, 2)])  # lon,lat format
        self.line2 = LineString([(0, 0), (1.1, 1.1), (2, 2)])

        # Create test GeoDataFrames
        self.df1 = gpd.GeoDataFrame(
            {"MMSI": [1234, 5678], "geometry": [self.line1, LineString([(0, 0), (1, 1)])]}
        )

        self.df2 = gpd.GeoDataFrame(
            {"MMSI": [1234, 5678], "geometry": [self.line2, LineString([(0, 0), (1.1, 1.1)])]}
        )

    def test_linestring_to_coords(self):
        """Test conversion from LineString to coordinate pairs."""
        coords = _linestring_to_coords(self.line1)

        # Test coordinate conversion and (lon,lat) to (lat,lon) swap
        self.assertEqual(len(coords), 3)
        self.assertEqual(coords[0], (0, 0))  # (lat, lon)
        self.assertEqual(coords[1], (1, 1))
        self.assertEqual(coords[2], (2, 2))

        # Test that coordinates are returned as list of tuples
        self.assertIsInstance(coords, list)
        self.assertIsInstance(coords[0], tuple)

    def test_calculate_dtw_distance(self):
        """Test DTW distance calculation."""
        coords1 = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]  # (lat, lon)
        coords2 = [(0.0, 0.0), (1.1, 1.1), (2.0, 2.0)]

        distance, avg_distance = _calculate_dtw_distance(coords1, coords2)

        # Test that distances are positive
        self.assertGreater(distance, 0)
        self.assertGreater(avg_distance, 0)

        # Test that average distance is less than total distance
        self.assertLess(avg_distance, distance)

        # Test identical trajectories
        distance, avg_distance = _calculate_dtw_distance(coords1, coords1)
        self.assertAlmostEqual(distance, 0, places=10)
        self.assertAlmostEqual(avg_distance, 0, places=10)

    def test_compare_trajectory_pairs(self):
        """Test comparison of trajectory pairs."""
        result = compare_trajectory_pairs(self.df1, self.df2)

        # Test output structure
        self.assertIsInstance(result, pd.DataFrame)
        expected_columns = {
            "mmsi",
            "trajectory_id",
            "dtw_distance",
            "avg_point_distance",
            "path_length_ratio",
            "original_points",
            "resampled_points",
        }
        self.assertEqual(set(result.columns), expected_columns)

        # Test number of results
        self.assertEqual(len(result), len(self.df1))

        # Test values
        self.assertEqual(result["original_points"].iloc[0], 3)
        self.assertEqual(result["resampled_points"].iloc[0], 3)
        self.assertEqual(result["path_length_ratio"].iloc[0], 0)  # Same number of points

    def test_compare_trajectory_pairs_validation(self):
        """Test input validation for trajectory comparison."""
        # Test different length DataFrames
        df_short = self.df1.iloc[0:1]
        with self.assertRaises(ValueError):
            compare_trajectory_pairs(df_short, self.df2)

        # Test MMSI mismatch
        df_wrong_mmsi = self.df2.copy()
        df_wrong_mmsi["MMSI"] = [9999, 9999]
        with self.assertRaises(AssertionError):
            compare_trajectory_pairs(self.df1, df_wrong_mmsi)

    def test_parallel_processing(self):
        """Test parallel processing functionality."""
        # Create larger test dataset
        df1_large = pd.concat([self.df1] * 5, ignore_index=True)
        df2_large = pd.concat([self.df2] * 5, ignore_index=True)

        # Compare results with and without parallel processing
        result_serial = compare_trajectory_pairs(df1_large, df2_large, n_processes=1)
        result_parallel = compare_trajectory_pairs(df1_large, df2_large, n_processes=2)

        print(result_serial)
        print(result_parallel)

        # Test that results are the same
        pd.testing.assert_frame_equal(
            result_serial.sort_values("trajectory_id").reset_index(drop=True),
            result_parallel.sort_values("trajectory_id").reset_index(drop=True),
            check_exact=False,  # Allow for small floating point differences
            rtol=1e-10,
        )
