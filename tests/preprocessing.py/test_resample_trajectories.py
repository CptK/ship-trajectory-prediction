from unittest import TestCase

import pandas as pd
from shapely.geometry import LineString

from prediction.preprocessing.resample_trajectories import (
    interpolate_circular,
    interpolate_point_on_line,
    process_single_trajectory,
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
        point = interpolate_point_on_line(line, 0.0)
        self.assertEqual((point.x, point.y), (0, 0))

        # Test end point
        point = interpolate_point_on_line(line, line.length)
        self.assertEqual((point.x, point.y), (1, 1))

        # Test midpoint
        point = interpolate_point_on_line(line, line.length / 2)
        self.assertAlmostEqual(point.x, 0.5)
        self.assertAlmostEqual(point.y, 0.5)

        # Test beyond line bounds
        point = interpolate_point_on_line(line, -1.0)
        self.assertEqual((point.x, point.y), (0, 0))
        point = interpolate_point_on_line(line, line.length + 1.0)
        self.assertEqual((point.x, point.y), (1, 1))

    def test_interpolate_circular(self):
        """Test circular interpolation for various angle combinations."""
        # Test simple interpolation
        result = interpolate_circular(0, 90, 0.5)
        self.assertEqual(result, 45)

        # Test wrap around 360
        result = interpolate_circular(350, 10, 0.5)
        self.assertEqual(result, 0)

        # Test negative angles
        result = interpolate_circular(-90, 90, 0.5)
        self.assertEqual(result, 180)

        # Test fractions other than 0.5
        result = interpolate_circular(0, 90, 0.25)
        self.assertEqual(result, 22.5)

        # Test full circle
        result = interpolate_circular(0, 360, 0.5)
        self.assertEqual(result, 0)

    def test_process_single_trajectory(self):
        """Test processing of a single trajectory with various checks."""
        result = process_single_trajectory(self.df.iloc[0], interval_seconds=180)  # 3-minute intervals

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
