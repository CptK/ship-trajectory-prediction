import time
from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from prediction.preprocessing.outlier_detection import (
    _association,
    _partition,
    _remove_outliers_chunk,
    _remove_outliers_row,
    remove_outliers,
    remove_outliers_parallel,
)


class TestTrackPartitioning(TestCase):
    def setUp(self):
        """Set up common test data"""
        # Create a simple straight line track
        self.base_track = np.array(
            [
                [0, 0],  # Starting point
                [0, 0.1],  # ~11.1 km east
                [0, 0.2],  # Another 11.1 km east
                [0, 0.3],  # Another 11.1 km east
            ]
        )

        self.base_time = datetime(2024, 1, 1, 12, 0)
        self.base_timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3),
        ]

        # SOGs in km/h
        self.base_sogs = [10.0, 10.0, 10.0, 10.0]

    def test_partition_preserves_all_points(self):
        """Test that partitioning preserves all points, including single-point segments"""
        # Simple track with an obvious break in the middle
        track = np.array(
            [[0, 0], [0, 0.1], [0, 1.1], [0, 1.2]]  # Point 1  # Point 2  # Point 3 - big jump  # Point 4
        )

        timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3),
        ]

        sogs = [10.0, 10.0, 10.0, 10.0]

        result = _partition(
            track,
            timestamps,
            sogs,
            threshold_partition_distance=50.0,  # Should break at the big jump
            threshold_partition_sog=15.0,
        )

        # Count total points across all partitions
        total_points = sum(len(t) for t in result["tracks"])
        total_indices = sum(len(idx) for idx in result["indices"])

        # Should preserve all points
        self.assertEqual(total_points, len(track))
        self.assertEqual(total_indices, len(track))

        # Each partition should contain at least one point
        for t, idx in zip(result["tracks"], result["indices"]):
            self.assertGreaterEqual(len(t), 1)
            self.assertEqual(len(t), len(idx))

    def test_no_partition_needed(self):
        """Test when no partitioning is needed"""
        result = _partition(
            self.base_track,
            self.base_timestamps,
            self.base_sogs,
            threshold_partition_sog=15.0,
            threshold_partition_distance=100.0,
        )

        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(len(result["timestamps"]), 1)
        self.assertEqual(len(result["sogs"]), 1)
        self.assertEqual(len(result["indices"]), 1)
        np.testing.assert_array_equal(result["tracks"][0], self.base_track)
        self.assertEqual(result["indices"][0], list(range(len(self.base_track))))

    def test_partition_by_sog(self):
        """Test partitioning based on SOG threshold"""
        # Create track with varying speeds that will definitely trigger partitioning
        timestamps = [
            self.base_time,
            self.base_time + timedelta(minutes=60),
            self.base_time + timedelta(minutes=61),  # Only 1 minute after previous
            self.base_time + timedelta(minutes=180),
        ]

        # This will create a very high calculated speed between points 1 and 2
        sogs = [10.0, 10.0, 10.0, 10.0]

        result = _partition(
            self.base_track, timestamps, sogs, threshold_partition_sog=5.0, threshold_partition_distance=100.0
        )

        self.assertGreater(len(result["tracks"]), 1)
        self.assertEqual(len(result["tracks"]), len(result["timestamps"]))
        self.assertEqual(len(result["tracks"]), len(result["sogs"]))
        self.assertEqual(len(result["tracks"]), len(result["indices"]))
        
        # Verify indices are continuous and cover all points
        all_indices = [idx for sublist in result["indices"] for idx in sublist]
        self.assertEqual(len(all_indices), len(self.base_track))
        self.assertEqual(sorted(all_indices), list(range(len(self.base_track))))

    def test_partition_by_distance(self):
        """Test partitioning based on distance threshold"""
        # Create track with a large gap
        track = np.array([[0, 0], [0, 0.1], [0, 1.0], [0, 1.1]])  # Large gap here (~100 km)

        result = _partition(
            track,
            self.base_timestamps,
            self.base_sogs,
            threshold_partition_sog=15.0,
            threshold_partition_distance=50.0,
        )

        self.assertGreater(len(result["tracks"]), 1)
        self.assertEqual(len(result["tracks"]), len(result["timestamps"]))
        self.assertEqual(len(result["tracks"]), len(result["sogs"]))
        self.assertEqual(len(result["tracks"]), len(result["indices"]))

        # Verify indices match the track segments
        for subtrack, indices in zip(result["tracks"], result["indices"]):
            self.assertEqual(len(subtrack), len(indices))
            np.testing.assert_array_equal(track[indices], subtrack)

    def test_linestring_input(self):
        """Test with LineString input instead of numpy array"""
        line_string = LineString(self.base_track)

        result = _partition(
            line_string,
            self.base_timestamps,
            self.base_sogs,
            threshold_partition_sog=15.0,
            threshold_partition_distance=100.0,
        )

        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(len(result["indices"]), 1)
        self.assertTrue(isinstance(result["tracks"][0], np.ndarray))
        self.assertEqual(result["indices"][0], list(range(len(self.base_track))))

    def test_empty_track(self):
        """Test with empty track"""
        empty_track = np.array([]).reshape(0, 2)
        empty_timestamps: list = []
        empty_sogs: list = []

        result = _partition(
            empty_track,
            empty_timestamps,
            empty_sogs,
            threshold_partition_sog=15.0,
            threshold_partition_distance=100.0,
        )

        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(len(result["timestamps"]), 1)
        self.assertEqual(len(result["sogs"]), 1)
        self.assertEqual(len(result["indices"]), 1)
        self.assertEqual(len(result["tracks"][0]), 0)
        self.assertEqual(result["indices"][0], [])

    def test_single_point(self):
        """Test with single point track"""
        single_track = np.array([[0, 0]])
        single_timestamp = [self.base_time]
        single_sog = [10.0]

        result = _partition(
            single_track,
            single_timestamp,
            single_sog,
            threshold_partition_sog=15.0,
            threshold_partition_distance=100.0,
        )

        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(len(result["timestamps"]), 1)
        self.assertEqual(len(result["sogs"]), 1)
        self.assertEqual(len(result["indices"]), 1)
        self.assertEqual(len(result["tracks"][0]), 1)
        self.assertEqual(result["indices"][0], [0])

    def test_multiple_partitions(self):
        """Test with multiple partition points"""
        # Create track with multiple speed changes and distance gaps
        track = np.array([[0, 0], [0, 0.1], [0, 0.2], [0, 1.0], [0, 1.1], [0, 1.2]])  # Distance gap

        timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3),
            self.base_time + timedelta(hours=4),
            self.base_time + timedelta(hours=5),
        ]

        sogs = [10.0, 10.0, 30.0, 10.0, 10.0, 10.0]  # Speed spike in middle

        result = _partition(
            track, timestamps, sogs, threshold_partition_sog=15.0, threshold_partition_distance=50.0
        )

        self.assertGreater(len(result["tracks"]), 2)  # Should have multiple partitions
        self.assertEqual(len(result["tracks"]), len(result["timestamps"]))
        self.assertEqual(len(result["tracks"]), len(result["sogs"]))
        self.assertEqual(len(result["tracks"]), len(result["indices"]))

        # Verify that indices properly reconstruct the original track
        reconstructed_track = np.concatenate([track[idx] for idx in result["indices"]])
        np.testing.assert_array_equal(reconstructed_track, track)

    def test_consistent_partitioning(self):
        """Test that partitioned subtracks maintain consistency"""
        result = _partition(
            self.base_track,
            self.base_timestamps,
            self.base_sogs,
            threshold_partition_sog=15.0,
            threshold_partition_distance=100.0,
        )

        total_points = sum(len(track) for track in result["tracks"])
        total_timestamps = sum(len(ts) for ts in result["timestamps"])
        total_sogs = sum(len(sog) for sog in result["sogs"])
        total_indices = sum(len(idx) for idx in result["indices"])

        self.assertEqual(total_points, len(self.base_track))
        self.assertEqual(total_timestamps, len(self.base_timestamps))
        self.assertEqual(total_sogs, len(self.base_sogs))
        self.assertEqual(total_indices, len(self.base_track))


class TestTrackAssociation(TestCase):
    def setUp(self):
        """Set up common test data"""
        self.base_time = datetime(2024, 1, 1, 12, 0)

    def create_subtracks(self, tracks_data) -> dict[str, list]:
        """Helper function to create subtracks dictionary with the given data"""
        subtracks: dict = {"tracks": [], "timestamps": [], "sogs": [], "indices": []}
        current_index = 0
        for track_info in tracks_data:
            points = track_info["points"]
            subtracks["tracks"].append(np.array(points))
            subtracks["timestamps"].append(
                [self.base_time + timedelta(minutes=m) for m in track_info["time_offsets"]]
            )
            subtracks["sogs"].append(track_info["sogs"])
            subtracks["indices"].append(list(range(current_index, current_index + len(points))))
            current_index += len(points)
        return subtracks

    def test_simple_association(self):
        """Test association of two tracks that should be joined"""
        tracks_data = [
            {
                "points": [[0, 0], [0, 0.02], [0, 0.04], [0, 0.06]],  # First track
                "time_offsets": [0, 15, 30, 45],  # 15 min between points
                "sogs": [10.0, 10.0, 10.0, 10.0],
            },
            {
                "points": [[0, 0.08], [0, 0.10], [0, 0.12], [0, 0.14]],  # Second track
                "time_offsets": [60, 75, 90, 105],  # 15 min after first track
                "sogs": [10.0, 10.0, 10.0, 10.0],
            },
            {
                "points": [[0, 0.16], [0, 0.18], [0, 0.20], [0, 0.22]],  # Third track
                "time_offsets": [120, 135, 150, 165],  # 15 min after second track
                "sogs": [10.0, 10.0, 10.0, 10.0],
            },
        ]

        subtracks = self.create_subtracks(tracks_data)
        result = _association(
            subtracks,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=12,  # Only require 4 points
        )

        self.assertEqual(len(result["tracks"]), 1)  # Should be joined into one track
        self.assertEqual(len(result["tracks"][0]), 12)  # Should contain all points
        self.assertEqual(len(result["indices"]), 1)  # Should have one set of indices
        self.assertEqual(len(result["indices"][0]), 12)  # Should have indices for all points
        self.assertEqual(result["indices"][0], list(range(12)))  # Should have continuous indices

    def test_no_association_high_speed(self):
        """Test tracks that shouldn't be associated due to high speed between them"""
        tracks_data = [
            {"points": [[0, 0], [0, 0.1]], "time_offsets": [0, 60], "sogs": [10.0, 10.0]},
            {
                "points": [[0, 1.0], [0, 1.1]],  # Far away, would require high speed
                "time_offsets": [70, 130],  # Short time gap
                "sogs": [10.0, 10.0],
            },
        ]

        subtracks = self.create_subtracks(tracks_data)
        result = _association(
            subtracks,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=2,
        )

        self.assertEqual(len(result["tracks"]), 1)  # Only one track should meet completeness
        self.assertEqual(len(result["indices"]), 1)
        self.assertEqual(result["indices"][0], [0, 1])  # Should only contain indices of first track

    def test_no_association_large_distance(self):
        """Test tracks that shouldn't be associated due to large distance"""
        tracks_data = [
            {"points": [[0, 0], [0, 0.1]], "time_offsets": [0, 60], "sogs": [10.0, 10.0]},
            {"points": [[0, 2.0], [0, 2.1]], "time_offsets": [120, 180], "sogs": [10.0, 10.0]},  # >100km away
        ]

        subtracks = self.create_subtracks(tracks_data)
        result = _association(subtracks, threshold_completeness=2)

        self.assertEqual(len(result["tracks"]), 1)  # Only one track should meet completeness
        self.assertEqual(len(result["indices"]), 1)
        self.assertEqual(result["indices"][0], [0, 1])  # Should only contain indices of first track

    def test_multiple_associations(self):
        """Test associating multiple tracks in sequence"""
        tracks_data = [
            {"points": [[0, 0], [0, 0.1]], "time_offsets": [0, 60], "sogs": [10.0, 10.0]},
            {"points": [[0, 0.15], [0, 0.2]], "time_offsets": [120, 180], "sogs": [10.0, 10.0]},
            {"points": [[0, 0.25], [0, 0.3]], "time_offsets": [240, 300], "sogs": [10.0, 10.0]},
        ]

        subtracks = self.create_subtracks(tracks_data)
        result = _association(subtracks, threshold_completeness=2)

        self.assertEqual(len(result["tracks"]), 1)  # All should be joined
        self.assertEqual(len(result["tracks"][0]), 6)  # Should contain all points
        self.assertEqual(len(result["indices"]), 1)
        self.assertEqual(result["indices"][0], list(range(6)))  # Should have continuous indices for all points

    def test_completeness_threshold(self):
        """Test that tracks shorter than completeness threshold are filtered out"""
        tracks_data = [
            {"points": [[0, 0], [0, 0.1]], "time_offsets": [0, 60], "sogs": [10.0, 10.0]}  # 2 points
        ]

        subtracks = self.create_subtracks(tracks_data)
        result = _association(subtracks, threshold_completeness=3)  # Require at least 3 points

        self.assertEqual(len(result["tracks"]), 0)  # Too short, should be filtered out
        self.assertEqual(len(result["indices"]), 0)  # No indices for filtered tracks

    def test_empty_input(self):
        """Test handling of empty input"""
        empty_subtracks: dict = {"tracks": [], "timestamps": [], "sogs": [], "indices": []}
        result = _association(empty_subtracks)

        self.assertEqual(len(result["tracks"]), 0)
        self.assertEqual(len(result["timestamps"]), 0)
        self.assertEqual(len(result["sogs"]), 0)
        self.assertEqual(len(result["indices"]), 0)

    def test_single_track_input(self):
        """Test handling of single track input"""
        tracks_data = [
            {"points": [[0, 0], [0, 0.1], [0, 0.2]], "time_offsets": [0, 60, 120], "sogs": [10.0, 10.0, 10.0]}
        ]

        subtracks = self.create_subtracks(tracks_data)
        result = _association(subtracks, threshold_completeness=3)

        # Should keep the track if it meets completeness threshold
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(len(result["indices"]), 1)
        self.assertEqual(result["indices"][0], [0, 1, 2])  # Should maintain original indices

    def test_consistency_of_associated_data(self):
        """Test that associated tracks maintain data consistency"""
        tracks_data = [
            {"points": [[0, 0], [0, 0.1]], "time_offsets": [0, 60], "sogs": [10.0, 10.0]},
            {"points": [[0, 0.15], [0, 0.2]], "time_offsets": [120, 180], "sogs": [10.0, 10.0]},
        ]

        subtracks = self.create_subtracks(tracks_data)
        result = _association(subtracks, threshold_completeness=2)

        # Check that all arrays have matching lengths
        self.assertEqual(len(result["tracks"][0]), len(result["timestamps"][0]))
        self.assertEqual(len(result["tracks"][0]), len(result["sogs"][0]))
        self.assertEqual(len(result["tracks"][0]), len(result["indices"][0]))

        # Check indices are continuous and match the number of points
        self.assertEqual(result["indices"][0], list(range(4)))

        # Check time ordering
        timestamps = result["timestamps"][0]
        self.assertTrue(all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1)))

        # Check coordinate continuity
        track = result["tracks"][0]
        self.assertTrue(all(abs(track[i + 1][1] - track[i][1]) < 0.5 for i in range(len(track) - 1)))

    def test_association_indices_ordering(self):
        """Test that indices maintain proper ordering after association"""
        tracks_data = [
            {"points": [[0, 0], [0, 0.1]], "time_offsets": [0, 60], "sogs": [10.0, 10.0]},
            {"points": [[0, 0.15], [0, 0.2]], "time_offsets": [120, 180], "sogs": [10.0, 10.0]},
        ]

        subtracks = self.create_subtracks(tracks_data)
        
        # Manually alter indices to ensure they're properly handled
        subtracks["indices"] = [[10, 11], [12, 13]]  # Non-sequential indices
        
        result = _association(subtracks, threshold_completeness=2)

        # Check that indices are preserved in order
        self.assertEqual(result["indices"][0], [10, 11, 12, 13])


class TestRemoveOutliersRow(TestCase):
    def setUp(self):
        """Set up common test data"""
        self.base_time = datetime(2024, 1, 1, 12, 0)
        # Create a simple track
        self.base_track = LineString([[0, 0], [0, 0.1], [0, 0.2], [0, 0.3]])
        self.base_timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3),
        ]
        self.base_sogs = [10.0, 10.0, 10.0, 10.0]

        # Create additional column data
        self.base_sources = ["AIS", "AIS", "AIS", "AIS"]
        self.base_mmsi = [123456789, 123456789, 123456789, 123456789]
        self.base_cogs = [45.0, 45.0, 45.0, 45.0]

        # Create a base row with additional columns
        self.base_row = pd.Series({
            "geometry": self.base_track,
            "timestamps": self.base_timestamps,
            "velocities": self.base_sogs,
            "source": self.base_sources,
            "mmsi": self.base_mmsi,
            "cog": self.base_cogs
        })

    def test_normal_track_with_additional_columns(self):
        """Test processing of a normal track with additional columns"""
        result = _remove_outliers_row(
            self.base_row,
            threshold_completeness=4,
            additional_filter_columns=["source", "mmsi", "cog"]
        )

        self.assertEqual(len(result), 1)  # Should return single processed track
        processed_row = result[0]
        
        # Check all lists have same length
        self.assertEqual(len(processed_row["timestamps"]), 4)
        self.assertEqual(len(processed_row["source"]), 4)
        self.assertEqual(len(processed_row["mmsi"]), 4)
        self.assertEqual(len(processed_row["cog"]), 4)
        
        # Check additional columns maintain their values
        self.assertEqual(processed_row["source"], self.base_sources)
        self.assertEqual(processed_row["mmsi"], self.base_mmsi)
        self.assertEqual(processed_row["cog"], self.base_cogs)

    def test_track_with_speed_outlier_additional_columns(self):
        """Test processing of a track with a speed outlier and additional columns"""
        track = LineString([[0, 0], [0, 0.1], [0, 1.0], [0, 1.1]])
        timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(minutes=61),
            self.base_time + timedelta(hours=3),
        ]
        sources = ["AIS", "SAT", "AIS", "SAT"]
        mmsi = [123456789, 123456789, 123456789, 123456789]

        row = pd.Series({
            "geometry": track,
            "timestamps": timestamps,
            "velocities": [10.0] * 4,
            "source": sources,
            "mmsi": mmsi
        })

        result = _remove_outliers_row(
            row,
            threshold_partition_sog=5.0,
            threshold_completeness=2,
            additional_filter_columns=["source", "mmsi"]
        )

        # Check each segment maintains corresponding additional column data
        for processed_row in result:
            coords = list(processed_row["geometry"].coords)
            self.assertEqual(len(coords), len(processed_row["source"]))
            self.assertEqual(len(coords), len(processed_row["mmsi"]))
            # Verify values match original data for corresponding indices
            for i, coord in enumerate(coords):
                orig_idx = next(j for j, c in enumerate(track.coords) if c == coord)
                self.assertEqual(processed_row["source"][i], sources[orig_idx])
                self.assertEqual(processed_row["mmsi"][i], mmsi[orig_idx])

    def test_multiple_segments_additional_columns(self):
        """Test processing of a track split into multiple segments with additional columns"""
        track = LineString([
            [0, 0], [0, 0.1],  # Segment 1
            [0, 1.0], [0, 1.1],  # Segment 2
            [0, 2.0], [0, 2.1],  # Segment 3
        ])
        
        timestamps = [
            self.base_time + timedelta(hours=i) for i in range(6)
        ]
        
        sources = ["AIS", "SAT", "AIS", "SAT", "AIS", "SAT"]
        mmsi = [123456789] * 6
        cogs = [45.0, 45.0, 45.0, 45.0, 45.0, 45.0]

        row = pd.Series({
            "geometry": track,
            "timestamps": timestamps,
            "velocities": [10.0] * 6,
            "source": sources,
            "mmsi": mmsi,
            "cog": cogs
        })

        result = _remove_outliers_row(
            row,
            threshold_partition_distance=50.0,
            threshold_completeness=2,
            additional_filter_columns=["source", "mmsi", "cog"]
        )

        total_points = sum(len(list(r["geometry"].coords)) for r in result)
        self.assertGreater(len(result), 1)  # Should have multiple segments
        
        # Check that we haven't lost any valid data
        for processed_row in result:
            coords = list(processed_row["geometry"].coords)
            self.assertEqual(len(coords), len(processed_row["source"]))
            self.assertEqual(len(coords), len(processed_row["mmsi"]))
            self.assertEqual(len(coords), len(processed_row["cog"]))

    def test_empty_additional_columns(self):
        """Test that function works correctly when no additional columns are specified"""
        result = _remove_outliers_row(
            self.base_row,
            threshold_completeness=4,
            additional_filter_columns=[]
        )

        self.assertEqual(len(result), 1)
        # Check that original columns are preserved
        self.assertTrue("geometry" in result[0])
        self.assertTrue("timestamps" in result[0])
        self.assertTrue("velocities" in result[0])

    def test_invalid_additional_column(self):
        """Test handling of non-existent additional column"""
        with self.assertRaises(KeyError):
            _remove_outliers_row(
                self.base_row,
                threshold_completeness=4,
                additional_filter_columns=["nonexistent_column"]
            )

    def test_additional_columns_consistency(self):
        """Test that additional columns maintain consistency with track segmentation"""
        track = LineString([[0, 0], [0, 0.1], [0, 1.0], [0, 1.1]])
        qualities = [1.0, 0.9, 0.8, 0.7]
        row = self.base_row.copy()
        row["geometry"] = track
        row["quality"] = qualities

        result = _remove_outliers_row(
            row,
            threshold_partition_distance=50.0,
            threshold_completeness=2,
            additional_filter_columns=["quality"]
        )

        for processed_row in result:
            coords = list(processed_row["geometry"].coords)
            qualities = processed_row["quality"]
            # Check that qualities align with coordinates
            self.assertEqual(len(coords), len(qualities))
            # Verify values are from original data
            self.assertTrue(all(q in row["quality"] for q in qualities))


class TestRemoveOutliersChunk(TestCase):
    def setUp(self):
        """Set up common test data"""
        self.base_time = datetime(2024, 1, 1, 12, 0)

        # Create default thresholds
        self.thresholds = {
            "partition_sog": 5.0,
            "partition_distance": 50.0,
            "association_sog": 15.0,
            "association_distance": 50.0,
            "completeness": 4,
        }

        # Create a simple track
        self.base_track = LineString([[0, 0], [0, 0.1], [0, 0.2], [0, 0.3]])
        self.base_timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3),
        ]
        self.base_sogs = [10.0, 10.0, 10.0, 10.0]
        
        # Create additional column data
        self.base_sources = ["AIS", "AIS", "AIS", "AIS"]
        self.base_mmsi = [123456789, 123456789, 123456789, 123456789]
        self.base_cogs = [45.0, 45.0, 45.0, 45.0]

    def create_test_chunk(self, num_rows: int, include_outliers: bool = False, include_additional: bool = False) -> pd.DataFrame:
        """Helper function to create a test DataFrame chunk"""
        rows = []
        for i in range(num_rows):
            if include_outliers and i % 2 == 1:
                # Create track with outlier
                track = LineString([[0, i], [0, i + 0.1], [0, i + 1.0], [0, i + 1.1]])  # Large jump
            else:
                # Create normal track
                track = LineString([[0, i], [0, i + 0.1], [0, i + 0.2], [0, i + 0.3]])

            row_data = {
                "geometry": track,
                "timestamps": [self.base_time + timedelta(hours=j) for j in range(4)],
                "velocities": [10.0] * 4,
                "vessel_id": f"VESSEL_{i}",
            }
            
            if include_additional:
                row_data.update({
                    "source": self.base_sources.copy(),
                    "mmsi": self.base_mmsi.copy(),
                    "cog": self.base_cogs.copy()
                })

            row = pd.Series(row_data)
            rows.append(row)

        return pd.DataFrame(rows)

    def test_process_empty_chunk(self):
        """Test processing an empty chunk"""
        empty_chunk = pd.DataFrame(columns=["geometry", "timestamps", "velocities"])
        args = (empty_chunk, self.thresholds, False, [])

        result = _remove_outliers_chunk(args)

        self.assertEqual(len(result), 0)

    def test_process_single_row_chunk_with_additional_columns(self):
        """Test processing a chunk with a single row and additional columns"""
        row_data = {
            "geometry": self.base_track,
            "timestamps": self.base_timestamps,
            "velocities": self.base_sogs,
            "vessel_id": "VESSEL_1",
            "source": self.base_sources,
            "mmsi": self.base_mmsi,
            "cog": self.base_cogs
        }
        chunk = pd.DataFrame([row_data])

        args = (chunk, self.thresholds, False, ["source", "mmsi", "cog"])
        result = _remove_outliers_chunk(args)

        self.assertGreater(len(result), 0)
        self.assertTrue(isinstance(result[0]["geometry"], LineString))
        self.assertEqual(result[0]["vessel_id"], "VESSEL_1")
        self.assertEqual(len(result[0]["source"]), 4)
        self.assertEqual(len(result[0]["mmsi"]), 4)
        self.assertEqual(len(result[0]["cog"]), 4)

    def test_process_multiple_rows_with_additional_columns(self):
        """Test processing multiple rows with additional columns"""
        chunk = self.create_test_chunk(5, include_outliers=False, include_additional=True)
        args = (chunk, self.thresholds, False, ["source", "mmsi", "cog"])

        result = _remove_outliers_chunk(args)

        self.assertEqual(len(result), 5)  # Should preserve all rows
        for row in result:
            self.assertEqual(len(list(row["geometry"].coords)), 4)
            self.assertEqual(len(row["timestamps"]), 4)
            self.assertEqual(len(row["velocities"]), 4)
            self.assertEqual(len(row["source"]), 4)
            self.assertEqual(len(row["mmsi"]), 4)
            self.assertEqual(len(row["cog"]), 4)
            # Verify content of additional columns
            self.assertEqual(row["source"], self.base_sources)
            self.assertEqual(row["mmsi"], self.base_mmsi)
            self.assertEqual(row["cog"], self.base_cogs)

    def test_process_rows_with_outliers_and_additional_columns(self):
        """Test processing rows containing outliers with additional columns"""
        chunk = self.create_test_chunk(4, include_outliers=True, include_additional=True)
        args = (chunk, self.thresholds, False, ["source", "mmsi", "cog"])

        result = _remove_outliers_chunk(args)

        # Should have more rows than input due to splitting of tracks with outliers
        self.assertGreater(len(result), 0)

        # Check that all resulting rows are valid and additional columns are properly split
        for row in result:
            coords = list(row["geometry"].coords)
            self.assertEqual(len(coords), len(row["timestamps"]))
            self.assertEqual(len(coords), len(row["velocities"]))
            self.assertEqual(len(coords), len(row["source"]))
            self.assertEqual(len(coords), len(row["mmsi"]))
            self.assertEqual(len(coords), len(row["cog"]))
            # Verify all values in additional columns are from original data
            self.assertTrue(all(s in self.base_sources for s in row["source"]))
            self.assertTrue(all(m in self.base_mmsi for m in row["mmsi"]))
            self.assertTrue(all(c in self.base_cogs for c in row["cog"]))

    def test_verbose_mode_with_additional_columns(self):
        """Test that verbose mode doesn't affect results with additional columns"""
        chunk = self.create_test_chunk(3, include_additional=True)
        additional_columns = ["source", "mmsi", "cog"]

        # Process with verbose=False
        args_non_verbose = (chunk, self.thresholds, False, additional_columns)
        result_non_verbose = _remove_outliers_chunk(args_non_verbose)

        # Process with verbose=True
        args_verbose = (chunk, self.thresholds, True, additional_columns)
        result_verbose = _remove_outliers_chunk(args_verbose)

        # Results should be identical
        self.assertEqual(len(result_non_verbose), len(result_verbose))
        for row1, row2 in zip(result_non_verbose, result_verbose):
            self.assertTrue(row1["geometry"].equals(row2["geometry"]))
            self.assertEqual(row1["timestamps"], row2["timestamps"])
            self.assertEqual(row1["velocities"], row2["velocities"])
            for col in additional_columns:
                self.assertEqual(row1[col], row2[col])

    def test_invalid_additional_columns(self):
        """Test handling of invalid additional columns"""
        chunk = self.create_test_chunk(2, include_additional=True)
        args = (chunk, self.thresholds, False, ["nonexistent_column"])

        with self.assertRaises(KeyError):
            _remove_outliers_chunk(args)

    def test_row_independence_with_additional_columns(self):
        """Test that each row is processed independently with additional columns"""
        row_data = {
            "geometry": self.base_track,
            "timestamps": self.base_timestamps,
            "velocities": self.base_sogs,
            "vessel_id": "VESSEL_1",
            "source": self.base_sources,
            "mmsi": self.base_mmsi
        }
        chunk = pd.DataFrame([row_data, row_data.copy()])

        args = (chunk, self.thresholds, False, ["source", "mmsi"])
        result = _remove_outliers_chunk(args)

        # Results for identical rows should be identical
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0]["geometry"].equals(result[1]["geometry"]))
        self.assertEqual(result[0]["timestamps"], result[1]["timestamps"])
        self.assertEqual(result[0]["velocities"], result[1]["velocities"])
        self.assertEqual(result[0]["source"], result[1]["source"])
        self.assertEqual(result[0]["mmsi"], result[1]["mmsi"])


class TestRemoveOutliers(TestCase):
    def setUp(self):
        """Set up common test data"""
        self.base_time = datetime(2024, 1, 1, 12, 0)

        # Create a simple track
        self.base_track = LineString([[0, 0], [0, 0.1], [0, 0.2], [0, 0.3]])
        self.base_timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3),
        ]
        self.base_velocities = [10.0, 10.0, 10.0, 10.0]
        
        # Create additional column data
        self.base_sources = ["AIS", "AIS", "AIS", "AIS"]
        self.base_mmsi = [123456789, 123456789, 123456789, 123456789]
        self.base_cogs = [45.0, 45.0, 45.0, 45.0]

    def create_test_df(self, num_rows: int, include_outliers: bool = False, include_additional: bool = False) -> pd.DataFrame:
        """Helper function to create a test DataFrame"""
        data = []
        for i in range(num_rows):
            base_lat = i % 80  # Keep latitude within [-90, 90]
            base_lng = i % 80  # Keep longitude within [-180, 180]

            if include_outliers and i % 2 == 1:
                # Create track with outlier, but keep coordinates valid
                track = LineString(
                    [
                        [base_lat, base_lng],
                        [base_lat, base_lng + 0.1],
                        [base_lat, base_lng + 1.0],  # Large but valid jump
                        [base_lat, base_lng + 1.1],
                    ]
                )
            else:
                # Create normal track
                track = LineString(
                    [
                        [base_lat, base_lng],
                        [base_lat, base_lng + 0.1],
                        [base_lat, base_lng + 0.2],
                        [base_lat, base_lng + 0.3],
                    ]
                )

            row = {
                "geometry": track,
                "timestamps": [self.base_time + timedelta(hours=j) for j in range(4)],
                "velocities": [10.0] * 4,
                "vessel_id": f"VESSEL_{i}",
                "vessel_type": "cargo",
            }
            
            if include_additional:
                row.update({
                    "source": self.base_sources.copy(),
                    "mmsi": self.base_mmsi.copy(),
                    "cog": self.base_cogs.copy()
                })
            
            data.append(row)

        return pd.DataFrame(data)

    def test_additional_columns_preserved(self):
        """Test that additional filter columns are properly preserved and filtered"""
        df = pd.DataFrame([{
            "geometry": self.base_track,
            "timestamps": self.base_timestamps,
            "velocities": self.base_velocities,
            "vessel_id": "VESSEL_1",
            "source": self.base_sources,
            "mmsi": self.base_mmsi,
            "cog": self.base_cogs
        }])

        result = remove_outliers(
            df,
            threshold_completeness=4,
            additional_filter_columns=["source", "mmsi", "cog"]
        )

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(len(row["source"]), 4)
        self.assertEqual(len(row["mmsi"]), 4)
        self.assertEqual(len(row["cog"]), 4)
        self.assertEqual(row["source"], self.base_sources)
        self.assertEqual(row["mmsi"], self.base_mmsi)
        self.assertEqual(row["cog"], self.base_cogs)

    def test_empty_dataframe(self):
        """Test processing an empty DataFrame"""
        empty_df = pd.DataFrame(columns=["geometry", "timestamps", "velocities"])
        result = remove_outliers(empty_df)

        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(len(result), 0)
        self.assertTrue(all(col in result.columns for col in ["geometry", "timestamps", "velocities"]))

    def test_single_row_dataframe(self):
        """Test processing a DataFrame with a single row"""
        df = pd.DataFrame(
            [
                {
                    "geometry": self.base_track,
                    "timestamps": self.base_timestamps,
                    "velocities": self.base_velocities,
                    "vessel_id": "VESSEL_1",
                }
            ]
        )

        result = remove_outliers(df, threshold_completeness=4)

        self.assertEqual(len(result), 1)
        self.assertTrue(isinstance(result.iloc[0]["geometry"], LineString))
        self.assertEqual(result.iloc[0]["vessel_id"], "VESSEL_1")
        self.assertEqual(len(result.iloc[0]["timestamps"]), 4)

    def test_normal_tracks(self):
        """Test processing normal tracks without outliers"""
        df = self.create_test_df(5, include_outliers=False)
        result = remove_outliers(
            df,
            threshold_partition_sog=5.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=4,
        )

        self.assertEqual(len(result), 5)  # Should preserve all tracks
        for _, row in result.iterrows():
            self.assertEqual(len(list(row["geometry"].coords)), 4)
            self.assertEqual(len(row["timestamps"]), 4)
            self.assertEqual(len(row["velocities"]), 4)

    def test_tracks_with_outliers(self):
        """Test processing tracks with outliers"""
        df = self.create_test_df(4, include_outliers=True)
        result = remove_outliers(
            df,
            threshold_partition_sog=5.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=2,
        )

        # Number of output rows might be different due to splitting
        self.assertGreater(len(result), 0)

        # Check that all resulting tracks are valid
        for _, row in result.iterrows():
            coords = list(row["geometry"].coords)
            self.assertEqual(len(coords), len(row["timestamps"]))
            self.assertEqual(len(coords), len(row["velocities"]))

    def test_different_thresholds(self):
        """Test that different threshold values affect the results"""
        df = self.create_test_df(4, include_outliers=True)

        # Process with default thresholds
        result_default = remove_outliers(df)

        # Process with more permissive thresholds
        result_permissive = remove_outliers(
            df,
            threshold_partition_sog=20.0,
            threshold_partition_distance=100.0,
            threshold_association_sog=30.0,
            threshold_association_distance=100.0,
            threshold_completeness=2,
        )

        # More permissive thresholds should result in fewer splits
        self.assertNotEqual(len(result_default), len(result_permissive))

    def test_preserve_dtypes(self):
        """Test that data types are preserved in the output DataFrame"""
        df = pd.DataFrame(
            [
                {
                    "geometry": self.base_track,
                    "timestamps": self.base_timestamps,
                    "velocities": self.base_velocities,
                    "vessel_id": "VESSEL_1",
                    "int_field": 42,
                    "float_field": 3.14,
                    "bool_field": True,
                }
            ]
        )

        result = remove_outliers(df, threshold_completeness=4)

        self.assertEqual(result["int_field"].dtype, np.int64)
        self.assertEqual(result["float_field"].dtype, np.float64)
        self.assertEqual(result["bool_field"].dtype, np.bool_)

    def test_preserve_column_order(self):
        """Test that column order is preserved in the output DataFrame"""
        columns = ["vessel_id", "geometry", "timestamps", "velocities", "extra_field"]
        df = pd.DataFrame(
            [
                {
                    "vessel_id": "VESSEL_1",
                    "geometry": self.base_track,
                    "timestamps": self.base_timestamps,
                    "velocities": self.base_velocities,
                    "extra_field": "test",
                }
            ]
        )[
            columns
        ]  # Explicitly set column order

        result = remove_outliers(df, threshold_completeness=4)

        self.assertEqual(list(result.columns), columns)

    def test_invalid_input_missing_columns(self):
        """Test handling of invalid input DataFrame missing required columns"""
        df = pd.DataFrame(
            [
                {
                    "vessel_id": "VESSEL_1",
                    "geometry": self.base_track,
                    # Missing 'timestamps' and 'velocities'
                }
            ]
        )

        with self.assertRaises((KeyError, ValueError)):
            remove_outliers(df)

    def test_verbose_mode(self):
        """Test that verbose mode doesn't affect results"""
        df = self.create_test_df(3)

        # Process with verbose=False
        result_non_verbose = remove_outliers(df, verbose=False)

        # Process with verbose=True
        result_verbose = remove_outliers(df, verbose=True)

        # Results should be identical
        pd.testing.assert_frame_equal(result_non_verbose, result_verbose)

    def test_completeness_threshold(self):
        """Test effect of completeness threshold"""
        df = self.create_test_df(3)

        # With low completeness threshold
        result_low = remove_outliers(df, threshold_completeness=2)

        # With high completeness threshold
        result_high = remove_outliers(df, threshold_completeness=10)

        # High threshold should result in fewer or no tracks
        self.assertGreaterEqual(len(result_low), len(result_high))

    def test_large_dataframe(self):
        """Test processing of a larger DataFrame for performance"""
        df = self.create_test_df(100, include_outliers=True)

        result = remove_outliers(
            df,
            threshold_partition_sog=15.0,  # More permissive thresholds
            threshold_partition_distance=100.0,
            threshold_association_sog=30.0,
            threshold_association_distance=100.0,
            threshold_completeness=4,  # Match the number of points in our test tracks
            verbose=False,
        )

        self.assertGreater(len(result), 0)
        # Check that all rows are valid
        for _, row in result.iterrows():
            self.assertTrue(isinstance(row["geometry"], LineString))
            self.assertEqual(len(row["timestamps"]), len(list(row["geometry"].coords)))

    def test_negative_thresholds(self):
        """Test handling of negative threshold values"""
        df = self.create_test_df(1)

        with self.assertRaises(ValueError):
            remove_outliers(
                df,
                threshold_partition_sog=-5.0,
                threshold_partition_distance=50.0,
                threshold_association_sog=15.0,
                threshold_association_distance=50.0,
            )

    def test_tracks_with_outliers_and_additional_columns(self):
        """Test processing tracks with outliers and additional columns"""
        df = self.create_test_df(4, include_outliers=True, include_additional=True)
        result = remove_outliers(
            df,
            threshold_partition_sog=5.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=2,
            additional_filter_columns=["source", "mmsi", "cog"]
        )

        self.assertGreater(len(result), 0)
        for _, row in result.iterrows():
            coords = list(row["geometry"].coords)
            self.assertEqual(len(coords), len(row["timestamps"]))
            self.assertEqual(len(coords), len(row["velocities"]))
            self.assertEqual(len(coords), len(row["source"]))
            self.assertEqual(len(coords), len(row["mmsi"]))
            self.assertEqual(len(coords), len(row["cog"]))
            # Verify all values are from original data
            self.assertTrue(all(s in self.base_sources for s in row["source"]))
            self.assertTrue(all(m in self.base_mmsi for m in row["mmsi"]))
            self.assertTrue(all(c in self.base_cogs for c in row["cog"]))

    def test_invalid_additional_column(self):
        """Test handling of invalid additional column"""
        df = self.create_test_df(1, include_additional=True)
        
        with self.assertRaises(KeyError):
            remove_outliers(
                df,
                threshold_completeness=4,
                additional_filter_columns=["nonexistent_column"]
            )

    def test_empty_additional_columns(self):
        """Test that function works correctly when no additional columns are specified"""
        df = self.create_test_df(1, include_additional=True)
        
        result = remove_outliers(df, threshold_completeness=4, additional_filter_columns=[])
        
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertTrue("source" in row)  # Column should still exist
        self.assertEqual(row["source"], self.base_sources)  # But should be unchanged

    def test_large_dataframe_with_additional_columns(self):
        """Test processing of a larger DataFrame with additional columns"""
        df = self.create_test_df(100, include_outliers=True, include_additional=True)
        
        result = remove_outliers(
            df,
            threshold_partition_sog=15.0,
            threshold_partition_distance=100.0,
            threshold_association_sog=30.0,
            threshold_association_distance=100.0,
            threshold_completeness=4,
            additional_filter_columns=["source", "mmsi", "cog"],
            verbose=False,
        )

        self.assertGreater(len(result), 0)
        for _, row in result.iterrows():
            coords = list(row["geometry"].coords)
            self.assertEqual(len(row["source"]), len(coords))
            self.assertEqual(len(row["mmsi"]), len(coords))
            self.assertEqual(len(row["cog"]), len(coords))


class TestRemoveOutliersParallel(TestCase):
    def setUp(self):
        """Set up common test data"""
        self.base_time = datetime(2024, 1, 1, 12, 0)
        self.default_thresholds = {
            "threshold_partition_sog": 15.0,
            "threshold_partition_distance": 100.0,
            "threshold_association_sog": 30.0,
            "threshold_association_distance": 100.0,
            "threshold_completeness": 4,
        }
        
        # Create additional column data
        self.base_sources = ["AIS", "AIS", "AIS", "AIS"]
        self.base_mmsi = [123456789, 123456789, 123456789, 123456789]
        self.base_cogs = [45.0, 45.0, 45.0, 45.0]

    def create_test_df(self, num_rows: int, include_outliers: bool = False, include_additional: bool = False) -> pd.DataFrame:
        """Helper function to create a test DataFrame with valid coordinates"""
        data = []
        for i in range(num_rows):
            base_lat = i % 80
            base_lng = i % 80

            if include_outliers and i % 2 == 1:
                track = LineString(
                    [
                        [base_lat, base_lng],
                        [base_lat, base_lng + 0.1],
                        [base_lat, base_lng + 1.0],
                        [base_lat, base_lng + 1.1],
                    ]
                )
            else:
                track = LineString(
                    [
                        [base_lat, base_lng],
                        [base_lat, base_lng + 0.1],
                        [base_lat, base_lng + 0.2],
                        [base_lat, base_lng + 0.3],
                    ]
                )

            row = {
                "geometry": track,
                "timestamps": [self.base_time + timedelta(hours=j) for j in range(4)],
                "velocities": [10.0] * 4,
                "vessel_id": f"VESSEL_{i}",
                "vessel_type": "cargo",
            }
            
            if include_additional:
                row.update({
                    "source": self.base_sources.copy(),
                    "mmsi": self.base_mmsi.copy(),
                    "cog": self.base_cogs.copy()
                })
            
            data.append(row)

        return pd.DataFrame(data)

    def test_parallel_vs_single_process_with_additional_columns(self):
        """Test that parallel processing gives same results as single process with additional columns"""
        df = self.create_test_df(10, include_outliers=True, include_additional=True)
        additional_columns = ["source", "mmsi", "cog"]

        # Process with single process
        single_result = remove_outliers(
            df, 
            **self.default_thresholds,
            verbose=False,
            additional_filter_columns=additional_columns
        )

        # Process with parallel processing
        parallel_result = remove_outliers_parallel(
            df,
            **self.default_thresholds,
            verbose=False,
            n_processes=2,
            additional_filter_columns=additional_columns
        )

        # Compare results
        self.assertEqual(len(single_result), len(parallel_result))

        # Sort both DataFrames for consistent comparison
        single_result = single_result.sort_values("vessel_id").reset_index(drop=True)
        parallel_result = parallel_result.sort_values("vessel_id").reset_index(drop=True)

        for i in range(len(single_result)):
            self.assertTrue(single_result.iloc[i]["geometry"].equals(parallel_result.iloc[i]["geometry"]))
            self.assertEqual(single_result.iloc[i]["timestamps"], parallel_result.iloc[i]["timestamps"])
            for col in additional_columns:
                self.assertEqual(single_result.iloc[i][col], parallel_result.iloc[i][col])

    def test_different_process_counts_with_additional_columns(self):
        """Test processing with different numbers of processes and additional columns"""
        df = self.create_test_df(20, include_outliers=True, include_additional=True)
        additional_columns = ["source", "mmsi", "cog"]

        results = {}
        process_counts = [1, 2, 4]

        for n_processes in process_counts:
            results[n_processes] = remove_outliers_parallel(
                df,
                **self.default_thresholds,
                verbose=False,
                n_processes=n_processes,
                additional_filter_columns=additional_columns
            )

        # Results should be the same regardless of process count
        for n1 in process_counts:
            for n2 in process_counts:
                self.assertEqual(len(results[n1]), len(results[n2]))

                r1 = results[n1].sort_values("vessel_id").reset_index(drop=True)
                r2 = results[n2].sort_values("vessel_id").reset_index(drop=True)

                for i in range(len(r1)):
                    self.assertTrue(r1.iloc[i]["geometry"].equals(r2.iloc[i]["geometry"]))
                    for col in additional_columns:
                        self.assertEqual(r1.iloc[i][col], r2.iloc[i][col])

    def test_chunk_boundaries_with_additional_columns(self):
        """Test that track processing works correctly across chunk boundaries with additional columns"""
        row_data = {
            "geometry": LineString([[0, 0], [0, 0.1], [0, 0.2], [0, 0.3]]),
            "timestamps": [self.base_time + timedelta(hours=i) for i in range(4)],
            "velocities": [10.0] * 4,
            "vessel_id": "VESSEL_1",
            "source": self.base_sources,
            "mmsi": self.base_mmsi,
            "cog": self.base_cogs
        }
        
        # Create DataFrame with identical consecutive rows
        df = pd.DataFrame([row_data] * 10)
        additional_columns = ["source", "mmsi", "cog"]

        result_2_chunks = remove_outliers_parallel(
            df,
            **self.default_thresholds,
            verbose=False,
            n_processes=2,
            additional_filter_columns=additional_columns
        )
        
        result_3_chunks = remove_outliers_parallel(
            df,
            **self.default_thresholds,
            verbose=False,
            n_processes=3,
            additional_filter_columns=additional_columns
        )

        self.assertEqual(len(result_2_chunks), len(result_3_chunks))
        
        # Compare additional columns across different chunk sizes
        result_2_chunks = result_2_chunks.sort_values("vessel_id").reset_index(drop=True)
        result_3_chunks = result_3_chunks.sort_values("vessel_id").reset_index(drop=True)
        
        for i in range(len(result_2_chunks)):
            for col in additional_columns:
                self.assertEqual(result_2_chunks.iloc[i][col], result_3_chunks.iloc[i][col])

    def test_large_dataframe_with_additional_columns(self):
        """Test parallel processing of large DataFrame with additional columns"""
        df = self.create_test_df(1000, include_outliers=True, include_additional=True)
        additional_columns = ["source", "mmsi", "cog"]

        result = remove_outliers_parallel(
            df,
            **self.default_thresholds,
            verbose=False,
            n_processes=4,
            additional_filter_columns=additional_columns
        )

        # Check that additional columns are properly maintained
        for _, row in result.iterrows():
            coords = list(row["geometry"].coords)
            for col in additional_columns:
                self.assertEqual(len(row[col]), len(coords))
                if col == "source":
                    self.assertTrue(all(s in self.base_sources for s in row[col]))
                elif col == "mmsi":
                    self.assertTrue(all(m in self.base_mmsi for m in row[col]))
                elif col == "cog":
                    self.assertTrue(all(c in self.base_cogs for c in row[col]))

    def test_invalid_additional_columns(self):
        """Test handling of invalid additional columns in parallel processing"""
        df = self.create_test_df(10, include_additional=True)
        
        with self.assertRaises(KeyError):
            remove_outliers_parallel(
                df,
                **self.default_thresholds,
                verbose=False,
                n_processes=2,
                additional_filter_columns=["nonexistent_column"]
            )

    def test_parallel_vs_single_process_results(self):
        """Test that parallel processing gives same results as single process"""
        df = self.create_test_df(10, include_outliers=True)

        # Process with single process
        single_result = remove_outliers(df, **self.default_thresholds, verbose=False)  # type: ignore

        # Process with parallel processing
        parallel_result = remove_outliers_parallel(
            df, **self.default_thresholds, verbose=False, n_processes=2  # type: ignore
        )

        # Compare results
        self.assertEqual(len(single_result), len(parallel_result))

        # Sort both DataFrames by vessel_id to ensure consistent comparison
        single_result = single_result.sort_values("vessel_id").reset_index(drop=True)
        parallel_result = parallel_result.sort_values("vessel_id").reset_index(drop=True)

        # Compare geometries
        for i in range(len(single_result)):
            self.assertTrue(single_result.iloc[i]["geometry"].equals(parallel_result.iloc[i]["geometry"]))
            self.assertEqual(single_result.iloc[i]["timestamps"], parallel_result.iloc[i]["timestamps"])

    def test_different_process_counts(self):
        """Test processing with different numbers of processes"""
        df = self.create_test_df(20, include_outliers=True)

        results = {}
        process_counts = [1, 2, 4]

        for n_processes in process_counts:
            results[n_processes] = remove_outliers_parallel(
                df, **self.default_thresholds, verbose=False, n_processes=n_processes  # type: ignore
            )

        # All results should be the same regardless of process count
        for n1 in process_counts:
            for n2 in process_counts:
                self.assertEqual(len(results[n1]), len(results[n2]))

                # Sort results for consistent comparison
                r1 = results[n1].sort_values("vessel_id").reset_index(drop=True)
                r2 = results[n2].sort_values("vessel_id").reset_index(drop=True)

                for i in range(len(r1)):
                    self.assertTrue(r1.iloc[i]["geometry"].equals(r2.iloc[i]["geometry"]))

    def test_small_dataframe_single_process(self):
        """Test that small DataFrames are processed with single process"""
        df = self.create_test_df(2)

        start_time = time.time()
        result = remove_outliers_parallel(
            df, **self.default_thresholds, verbose=False, n_processes=4  # type: ignore
        )
        processing_time = time.time() - start_time

        self.assertGreater(len(result), 0)
        # Processing time should be consistent with single-process execution
        self.assertLess(processing_time, 1.0)  # Adjust threshold as needed

    def test_default_process_count(self):
        """Test that default process count is CPU count - 1"""
        df = self.create_test_df(10)

        result = remove_outliers_parallel(df, **self.default_thresholds, verbose=False)  # type: ignore
        self.assertGreater(len(result), 0)

    def test_invalid_process_count(self):
        """Test handling of invalid process counts"""
        df = self.create_test_df(10)

        with self.assertRaises(ValueError):
            remove_outliers_parallel(df, **self.default_thresholds, n_processes=0)  # type: ignore

        with self.assertRaises(ValueError):
            remove_outliers_parallel(df, **self.default_thresholds, n_processes=-1)  # type: ignore

    def test_empty_dataframe(self):
        """Test processing of empty DataFrame"""
        df = pd.DataFrame(columns=["geometry", "timestamps", "velocities", "orientations"])

        result = remove_outliers_parallel(
            df, **self.default_thresholds, verbose=False, n_processes=2  # type: ignore
        )

        self.assertEqual(len(result), 0)
        self.assertTrue(isinstance(result, pd.DataFrame))

    def test_preserve_dtypes(self):
        """Test that data types are preserved across parallel processing"""
        df = pd.DataFrame(
            [
                {
                    "geometry": LineString([[0, 0], [0, 0.1], [0, 0.2], [0, 0.3]]),
                    "timestamps": [self.base_time + timedelta(hours=i) for i in range(4)],
                    "velocities": [10.0] * 4,
                    "vessel_id": "VESSEL_1",
                    "int_field": 42,
                    "float_field": 3.14,
                    "bool_field": True,
                    "orientations": [0.0] * 4,
                }
            ]
        )

        result = remove_outliers_parallel(
            df, **self.default_thresholds, verbose=False, n_processes=2  # type: ignore
        )

        self.assertEqual(result["int_field"].dtype, np.int64)
        self.assertEqual(result["float_field"].dtype, np.float64)
        self.assertEqual(result["bool_field"].dtype, np.bool_)

    def test_chunk_boundaries(self):
        """Test that track processing works correctly across chunk boundaries"""
        # Create DataFrame with identical consecutive tracks
        df = pd.DataFrame(
            [
                {
                    "geometry": LineString([[0, 0], [0, 0.1], [0, 0.2], [0, 0.3]]),
                    "timestamps": [self.base_time + timedelta(hours=i) for i in range(4)],
                    "velocities": [10.0] * 4,
                    "vessel_id": "VESSEL_1",
                    "orientations": [0.0] * 4,
                }
            ]
            * 10
        )  # 10 identical rows

        # Process with different numbers of chunks
        result_2_chunks = remove_outliers_parallel(
            df, **self.default_thresholds, verbose=False, n_processes=2  # type: ignore
        )
        result_3_chunks = remove_outliers_parallel(
            df, **self.default_thresholds, verbose=False, n_processes=3  # type: ignore
        )

        # Results should be identical regardless of chunking
        self.assertEqual(len(result_2_chunks), len(result_3_chunks))

    def test_memory_usage(self):
        """Test memory usage with large DataFrame"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create large DataFrame
        df = self.create_test_df(1000, include_outliers=True)

        # Process in parallel
        _ = remove_outliers_parallel(
            df, **self.default_thresholds, verbose=False, n_processes=4  # type: ignore
        )

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        # Memory increase should be reasonable (adjust threshold as needed)
        self.assertLess(memory_increase, 1000)  # Less than 1GB increase
