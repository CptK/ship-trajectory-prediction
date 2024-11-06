from unittest import TestCase

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from prediction.preprocessing.ais_status_segmentation import (
    _any_has_type_linestring,
    _get_segments,
    _process_row,
    _process_rows,
    _validate_input,
    segment_by_status,
)


class TestStatusSegmentation(TestCase):
    def setUp(self):
        # Basic test case - ensure LineStrings have at least 2 points
        self.df = pd.DataFrame(
            {
                "geometry": [LineString([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])],
                "orientations": [[1, 2, 3, 4, 5, 6]],
                "velocities": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
                "timestamps": [[0, 1, 2, 3, 4, 5]],
                "status": [[1, 1, 1, 2, 1, 1]],
                "point_count": [6],
                "unchanged": ["A"],
            }
        )

        # Multiple rows test case - ensure each segment has at least 2 points
        self.multi_df = pd.DataFrame(
            {
                "geometry": [
                    LineString([(0, 0), (1, 1), (2, 2)]),
                    LineString([(3, 3), (4, 4), (5, 5), (6, 6)]),
                ],
                "orientations": [[1, 2, 3], [4, 5, 6, 7]],
                "velocities": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]],
                "timestamps": [[0, 1, 2], [3, 4, 5, 6]],
                "status": [[1, 1, 2], [1, 1, 2, 1]],
                "point_count": [3, 4],
                "unchanged": ["A", "B"],
            }
        )

    def test_validate_input_basic(self):
        """Test basic input validation"""
        _validate_input(
            df=self.df,
            status_column="status",
            min_segment_length=None,
            additional_list_columns=["geometry", "orientations"],
            point_count_column="point_count",
            n_processes=1,
        )

    def test_validate_input_errors(self):
        """Test input validation error cases"""
        # Invalid status column
        with self.assertRaises(ValueError):
            _validate_input(
                df=self.df,
                status_column="invalid_column",
                min_segment_length=None,
                additional_list_columns=["geometry"],
                point_count_column="point_count",
                n_processes=1,
            )

        # Invalid min_segment_length
        with self.assertRaises(ValueError):
            _validate_input(
                df=self.df,
                status_column="status",
                min_segment_length=-1,
                additional_list_columns=["geometry"],
                point_count_column="point_count",
                n_processes=1,
            )

        # Invalid n_processes
        with self.assertRaises(ValueError):
            _validate_input(
                df=self.df,
                status_column="status",
                min_segment_length=None,
                additional_list_columns=["geometry"],
                point_count_column="point_count",
                n_processes=-1,
            )

    def test_get_segments_basic(self):
        """Test basic segment creation"""
        segments = _get_segments(statuses=[1, 1, 1, 2, 1, 1], split_statuses=[2], min_segment_length=None)
        expected = [[0, 1, 2], [4, 5]]
        self.assertEqual(segments, expected)

    def test_get_segments_edge_cases(self):
        """Test segment creation edge cases"""
        # No split points
        segments = _get_segments(statuses=[1, 1, 1], split_statuses=[2], min_segment_length=None)
        self.assertEqual(segments, [[0, 1, 2]])

        # All split points
        segments = _get_segments(statuses=[2, 2, 2], split_statuses=[2], min_segment_length=None)
        self.assertEqual(segments, [])

        # Min segment length filtering
        segments = _get_segments(statuses=[1, 1, 2, 1], split_statuses=[2], min_segment_length=3)
        self.assertEqual(segments, [])

    def test_process_row_basic(self):
        """Test basic row processing"""
        row = self.df.iloc[0]
        result = _process_row(
            row=row,
            status_column="status",
            default_status=1,
            split_statuses=[2],
            min_segment_length=None,
            additional_list_columns=["geometry", "orientations", "velocities", "timestamps"],
            point_count_column="point_count",
        )

        self.assertEqual(len(result), 2)  # Should create 2 segments
        self.assertEqual(result[0]["point_count"], 3)
        self.assertEqual(result[1]["point_count"], 2)

    def test_process_row_with_nan(self):
        """Test row processing with NaN values"""
        df_with_nan = pd.DataFrame(
            {
                "geometry": [LineString([(0, 0), (1, 1), (2, 2)])],
                "status": [[np.nan, 1, 2]],
                "point_count": [3],
            }
        )

        row = df_with_nan.iloc[0]
        result = _process_row(
            row=row,
            status_column="status",
            default_status=1,
            split_statuses=[2],
            min_segment_length=None,
            additional_list_columns=["geometry"],
            point_count_column="point_count",
        )

        self.assertEqual(result[0]["status"], [1, 1])  # NaN should be replaced with default_status

    def test_any_has_type_linestring(self):
        """Test LineString type detection"""
        # Test with LineString
        self.assertTrue(_any_has_type_linestring(self.df, ["geometry"]))
        # Test without LineString
        self.assertFalse(_any_has_type_linestring(self.df, ["orientations"]))
        # Test empty DataFrame
        empty_df = pd.DataFrame({"geometry": []})
        self.assertFalse(_any_has_type_linestring(empty_df, ["geometry"]))

    def test_process_rows_empty(self):
        """Test processing empty DataFrame"""
        empty_df = pd.DataFrame({"geometry": [], "status": [], "point_count": []})

        result = _process_rows(
            rows=empty_df,
            status_column="status",
            default_status=1,
            split_statuses=[2],
            min_segment_length=None,
            additional_list_columns=["geometry"],
            point_count_column="point_count",
        )

        self.assertTrue(result.empty)

    def test_min_segment_length_with_linestring(self):
        """Test that min_segment_length is automatically set to 2 with LineStrings"""
        # Test with None
        result = segment_by_status(
            df=self.df, status_column="status", default_status=1, split_statuses=[2], min_segment_length=None
        )
        self.assertTrue(all(len(row["geometry"].coords) >= 2 for _, row in result.iterrows()))

        # Test with min_segment_length < 2
        result = segment_by_status(
            df=self.df, status_column="status", default_status=1, split_statuses=[2], min_segment_length=1
        )
        self.assertTrue(all(len(row["geometry"].coords) >= 2 for _, row in result.iterrows()))

    def test_parallel_processing(self):
        """Test parallel processing with multiple rows"""
        # Test with 2 processes
        result = segment_by_status(
            df=self.multi_df, status_column="status", default_status=1, split_statuses=[2], n_processes=2
        )

        # Verify results
        self.assertTrue(all(len(row["geometry"].coords) >= 2 for _, row in result.iterrows()))

        # Test with automatic process count
        result_auto = segment_by_status(
            df=self.multi_df, status_column="status", default_status=1, split_statuses=[2], n_processes=None
        )

        # Results should be the same regardless of process count
        pd.testing.assert_frame_equal(result.reset_index(drop=True), result_auto.reset_index(drop=True))

    def test_parallel_processing_chunks(self):
        """Test that parallel processing creates appropriate chunks and processes"""
        # Test with different DataFrame sizes and process counts
        test_cases = [
            (self.df, 2),  # Single row DF with 2 processes
            (self.multi_df, 4),  # Multi row DF with 4 processes
            (self.df, 10),  # More processes than rows
        ]

        for test_df, n_processes in test_cases:
            result = segment_by_status(
                df=test_df,
                status_column="status",
                default_status=1,
                split_statuses=[2],
                n_processes=n_processes,
            )

            # Verify the result is valid
            self.assertGreater(len(result), 0)
            if "geometry" in result.columns:
                self.assertTrue(all(len(row["geometry"].coords) >= 2 for _, row in result.iterrows()))

            # Verify results are consistent with single-process version
            single_process_result = segment_by_status(
                df=test_df, status_column="status", default_status=1, split_statuses=[2], n_processes=1
            )

            pd.testing.assert_frame_equal(
                result.reset_index(drop=True), single_process_result.reset_index(drop=True)
            )

    def test_segment_by_status_with_min_length(self):
        """Test segmentation with minimum length requirement"""
        result = segment_by_status(
            df=self.df, status_column="status", default_status=1, split_statuses=[2], min_segment_length=3
        )

        # Should only keep segments with length >= 3
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["point_count"], 3)

    def test_segment_by_status_multiple_split_statuses(self):
        """Test segmentation with multiple split statuses"""
        df = pd.DataFrame(
            {
                "geometry": [LineString([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])],
                "status": [[1, 1, 2, 3, 1, 1]],
                "point_count": [6],
            }
        )

        result = segment_by_status(
            df=df,
            status_column="status",
            default_status=1,
            split_statuses=[2, 3],
            additional_list_columns=["geometry"],
        )

        self.assertEqual(len(result), 2)  # Should create 2 segments despite having 2 split statuses

    def test_unchanged_columns(self):
        """Test that non-list columns remain unchanged"""
        result = segment_by_status(
            df=self.df,
            status_column="status",
            default_status=1,
            split_statuses=[2],
            additional_list_columns=["geometry", "orientations"],
        )

        # Check that 'unchanged' column values are preserved
        self.assertTrue(all(val == "A" for val in result["unchanged"]))
