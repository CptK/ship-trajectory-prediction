from unittest import TestCase
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from shapely.geometry import LineString
import time

from prediction.preprocessing.outlier_detection import _partition, _association, _remove_outliers_row, _remove_outliers_chunk
from prediction.preprocessing.outlier_detection import remove_outliers, remove_outliers_parallel


class TestTrackPartitioning(TestCase):
    def setUp(self):
        """Set up common test data"""
        # Create a simple straight line track
        self.base_track = np.array([
            [0, 0],  # Starting point
            [0, 0.1],  # ~11.1 km east
            [0, 0.2],  # Another 11.1 km east
            [0, 0.3],  # Another 11.1 km east
        ])
        
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
        track = np.array([
            [0, 0],     # Point 1
            [0, 0.1],   # Point 2
            [0, 1.1],   # Point 3 - big jump
            [0, 1.2]    # Point 4
        ])
        
        timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3)
        ]
        
        sogs = [10.0, 10.0, 10.0, 10.0]
        
        result = _partition(
            track,
            timestamps,
            sogs,
            threshold_partition_distance=50.0,  # Should break at the big jump
            threshold_partition_sog=15.0
        )
        
        # Count total points across all partitions
        total_points = sum(len(t) for t in result["tracks"])

        # Should preserve all points
        self.assertEqual(total_points, len(track))
        
        # Each partition should contain at least one point
        for t in result["tracks"]:
            self.assertGreaterEqual(len(t), 1)

    def test_no_partition_needed(self):
        """Test when no partitioning is needed"""
        result = _partition(
            self.base_track,
            self.base_timestamps,
            self.base_sogs,
            threshold_partition_sog=15.0,
            threshold_partition_distance=100.0
        )
        
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(len(result["timestamps"]), 1)
        self.assertEqual(len(result["sogs"]), 1)
        np.testing.assert_array_equal(result["tracks"][0], self.base_track)

    def test_partition_by_sog(self):
        """Test partitioning based on SOG threshold"""
        # Create track with varying speeds that will definitely trigger partitioning
        timestamps = [
            self.base_time,
            self.base_time + timedelta(minutes=60),
            self.base_time + timedelta(minutes=61),  # Only 1 minute after previous
            self.base_time + timedelta(minutes=180)
        ]
        
        # This will create a very high calculated speed between points 1 and 2
        sogs = [10.0, 10.0, 10.0, 10.0]
        
        result = _partition(
            self.base_track,
            timestamps,
            sogs,
            threshold_partition_sog=5.0,
            threshold_partition_distance=100.0
        )
        print(result)
        
        self.assertGreater(len(result["tracks"]), 1)
        self.assertEqual(len(result["tracks"]), len(result["timestamps"]))
        self.assertEqual(len(result["tracks"]), len(result["sogs"]))

    def test_partition_by_distance(self):
        """Test partitioning based on distance threshold"""
        # Create track with a large gap
        track = np.array([
            [0, 0],
            [0, 0.1],
            [0, 1.0],  # Large gap here (~100 km)
            [0, 1.1]
        ])
        
        result = _partition(
            track,
            self.base_timestamps,
            self.base_sogs,
            threshold_partition_sog=15.0,
            threshold_partition_distance=50.0
        )
        
        self.assertGreater(len(result["tracks"]), 1)
        self.assertEqual(len(result["tracks"]), len(result["timestamps"]))
        self.assertEqual(len(result["tracks"]), len(result["sogs"]))

    def test_linestring_input(self):
        """Test with LineString input instead of numpy array"""
        line_string = LineString(self.base_track)
        
        result = _partition(
            line_string,
            self.base_timestamps,
            self.base_sogs,
            threshold_partition_sog=15.0,
            threshold_partition_distance=100.0
        )
        
        self.assertEqual(len(result["tracks"]), 1)
        self.assertTrue(isinstance(result["tracks"][0], np.ndarray))

    def test_length_mismatch(self):
        """Test that mismatched lengths raise an assertion error"""
        with self.assertRaises(AssertionError):
            _partition(
                self.base_track,
                self.base_timestamps[:-1],  # One less timestamp
                self.base_sogs,
                threshold_partition_sog=15.0,
                threshold_partition_distance=100.0
            )

    def test_empty_track(self):
        """Test with empty track"""
        empty_track = np.array([]).reshape(0, 2)
        empty_timestamps = []
        empty_sogs = []
        
        result = _partition(
            empty_track,
            empty_timestamps,
            empty_sogs,
            threshold_partition_sog=15.0,
            threshold_partition_distance=100.0
        )
        
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(len(result["timestamps"]), 1)
        self.assertEqual(len(result["sogs"]), 1)
        self.assertEqual(len(result["tracks"][0]), 0)

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
            threshold_partition_distance=100.0
        )
        
        self.assertEqual(len(result["tracks"]), 1)
        self.assertEqual(len(result["timestamps"]), 1)
        self.assertEqual(len(result["sogs"]), 1)
        self.assertEqual(len(result["tracks"][0]), 1)

    def test_multiple_partitions(self):
        """Test with multiple partition points"""
        # Create track with multiple speed changes and distance gaps
        track = np.array([
            [0, 0],
            [0, 0.1],
            [0, 0.2],
            [0, 1.0],  # Distance gap
            [0, 1.1],
            [0, 1.2]
        ])
        
        timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3),
            self.base_time + timedelta(hours=4),
            self.base_time + timedelta(hours=5)
        ]
        
        sogs = [10.0, 10.0, 30.0, 10.0, 10.0, 10.0]  # Speed spike in middle
        
        result = _partition(
            track,
            timestamps,
            sogs,
            threshold_partition_sog=15.0,
            threshold_partition_distance=50.0
        )
        
        self.assertGreater(len(result["tracks"]), 2)  # Should have multiple partitions
        self.assertEqual(len(result["tracks"]), len(result["timestamps"]))
        self.assertEqual(len(result["tracks"]), len(result["sogs"]))

    def test_consistent_partitioning(self):
        """Test that partitioned subtracks maintain consistency"""
        result = _partition(
            self.base_track,
            self.base_timestamps,
            self.base_sogs,
            threshold_partition_sog=15.0,
            threshold_partition_distance=100.0
        )
        
        total_points = sum(len(track) for track in result["tracks"])
        total_timestamps = sum(len(ts) for ts in result["timestamps"])
        total_sogs = sum(len(sog) for sog in result["sogs"])
        
        self.assertEqual(total_points, len(self.base_track))
        self.assertEqual(total_timestamps, len(self.base_timestamps))
        self.assertEqual(total_sogs, len(self.base_sogs))


class TestTrackAssociation(TestCase):
    def setUp(self):
        """Set up common test data"""
        self.base_time = datetime(2024, 1, 1, 12, 0)

    def create_subtracks(self, tracks_data) -> dict[str, list]:
        """Helper function to create subtracks dictionary with the given data"""
        subtracks = {"tracks": [], "timestamps": [], "sogs": []}
        for track_info in tracks_data:
            subtracks["tracks"].append(np.array(track_info["points"]))
            subtracks["timestamps"].append([
                self.base_time + timedelta(minutes=m) for m in track_info["time_offsets"]
            ])
            subtracks["sogs"].append(track_info["sogs"])
        return subtracks

    def test_simple_association(self):
        """Test association of two tracks that should be joined"""
        # Create tracks that should be associated
        tracks_data = [
            {
                "points": [[0, 0], [0, 0.02], [0, 0.04], [0, 0.06]],  # First track
                "time_offsets": [0, 15, 30, 45],       # 15 min between points
                "sogs": [10.0, 10.0, 10.0, 10.0]
            },
            {
                "points": [[0, 0.08], [0, 0.10], [0, 0.12], [0, 0.14]],  # Second track
                "time_offsets": [60, 75, 90, 105],     # 15 min after first track
                "sogs": [10.0, 10.0, 10.0, 10.0]
            },
            {
                "points": [[0, 0.16], [0, 0.18], [0, 0.20], [0, 0.22]],  # Third track
                "time_offsets": [120, 135, 150, 165],  # 15 min after second track
                "sogs": [10.0, 10.0, 10.0, 10.0]
            }
        ]
        
        subtracks = self.create_subtracks(tracks_data)
        result = _association(
            subtracks,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=12  # Only require 4 points
        )
        
        self.assertEqual(len(result["tracks"]), 1)  # Should be joined into one track
        self.assertEqual(len(result["tracks"][0]), 12)  # Should contain all points

    def test_no_association_high_speed(self):
        """Test tracks that shouldn't be associated due to high speed between them"""
        tracks_data = [
            {
                "points": [[0, 0], [0, 0.1]],
                "time_offsets": [0, 60],
                "sogs": [10.0, 10.0]
            },
            {
                "points": [[0, 1.0], [0, 1.1]],  # Far away, would require high speed
                "time_offsets": [70, 130],        # Short time gap
                "sogs": [10.0, 10.0]
            }
        ]
        
        subtracks = self.create_subtracks(tracks_data)
        result = _association(
            subtracks,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=2
        )
        
        self.assertEqual(len(result["tracks"]), 1)  # Only one track should meet completeness
        
    def test_no_association_large_distance(self):
        """Test tracks that shouldn't be associated due to large distance"""
        tracks_data = [
            {
                "points": [[0, 0], [0, 0.1]],
                "time_offsets": [0, 60],
                "sogs": [10.0, 10.0]
            },
            {
                "points": [[0, 2.0], [0, 2.1]],  # >100km away
                "time_offsets": [120, 180],
                "sogs": [10.0, 10.0]
            }
        ]
        
        subtracks = self.create_subtracks(tracks_data)
        result = _association(subtracks, threshold_completeness=2)
        
        self.assertEqual(len(result["tracks"]), 1)  # Only one track should meet completeness

    def test_multiple_associations(self):
        """Test associating multiple tracks in sequence"""
        tracks_data = [
            {
                "points": [[0, 0], [0, 0.1]],
                "time_offsets": [0, 60],
                "sogs": [10.0, 10.0]
            },
            {
                "points": [[0, 0.15], [0, 0.2]],
                "time_offsets": [120, 180],
                "sogs": [10.0, 10.0]
            },
            {
                "points": [[0, 0.25], [0, 0.3]],
                "time_offsets": [240, 300],
                "sogs": [10.0, 10.0]
            }
        ]
        
        subtracks = self.create_subtracks(tracks_data)
        result = _association(subtracks, threshold_completeness=2)
        
        self.assertEqual(len(result["tracks"]), 1)  # All should be joined
        self.assertEqual(len(result["tracks"][0]), 6)  # Should contain all points

    def test_completeness_threshold(self):
        """Test that tracks shorter than completeness threshold are filtered out"""
        tracks_data = [
            {
                "points": [[0, 0], [0, 0.1]],  # 2 points
                "time_offsets": [0, 60],
                "sogs": [10.0, 10.0]
            }
        ]
        
        subtracks = self.create_subtracks(tracks_data)
        result = _association(
            subtracks,
            threshold_completeness=3  # Require at least 3 points
        )
        
        self.assertEqual(len(result["tracks"]), 0)  # Too short, should be filtered out

    def test_empty_input(self):
        """Test handling of empty input"""
        empty_subtracks = {"tracks": [], "timestamps": [], "sogs": []}
        result = _association(empty_subtracks)
        
        self.assertEqual(len(result["tracks"]), 0)
        self.assertEqual(len(result["timestamps"]), 0)
        self.assertEqual(len(result["sogs"]), 0)

    def test_single_track_input(self):
        """Test handling of single track input"""
        tracks_data = [
            {
                "points": [[0, 0], [0, 0.1], [0, 0.2]],
                "time_offsets": [0, 60, 120],
                "sogs": [10.0, 10.0, 10.0]
            }
        ]
        
        subtracks = self.create_subtracks(tracks_data)
        result = _association(subtracks, threshold_completeness=3)
        
        # Should keep the track if it meets completeness threshold
        self.assertEqual(len(result["tracks"]), 1)

    def test_consistency_of_associated_data(self):
        """Test that associated tracks maintain data consistency"""
        tracks_data = [
            {
                "points": [[0, 0], [0, 0.1]],
                "time_offsets": [0, 60],
                "sogs": [10.0, 10.0]
            },
            {
                "points": [[0, 0.15], [0, 0.2]],
                "time_offsets": [120, 180],
                "sogs": [10.0, 10.0]
            }
        ]
        
        subtracks = self.create_subtracks(tracks_data)
        result = _association(subtracks, threshold_completeness=2)
        
        # Check that all arrays have matching lengths
        self.assertEqual(len(result["tracks"][0]), len(result["timestamps"][0]))
        self.assertEqual(len(result["tracks"][0]), len(result["sogs"][0]))
        
        # Check time ordering
        timestamps = result["timestamps"][0]
        self.assertTrue(all(timestamps[i] < timestamps[i+1] 
                          for i in range(len(timestamps)-1)))

        # Check coordinate continuity
        track = result["tracks"][0]
        self.assertTrue(all(abs(track[i+1][1] - track[i][1]) < 0.5 
                          for i in range(len(track)-1)))


class TestRemoveOutliersRow(TestCase):
    def setUp(self):
        """Set up common test data"""
        self.base_time = datetime(2024, 1, 1, 12, 0)
        # Create a simple track
        self.base_track = LineString([
            [0, 0],
            [0, 0.1],
            [0, 0.2],
            [0, 0.3]
        ])
        self.base_timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3)
        ]
        self.base_sogs = [10.0, 10.0, 10.0, 10.0]
        
        # Create a base row
        self.base_row = pd.Series({
            'geometry': self.base_track,
            'timestamps': self.base_timestamps,
            'velocities': self.base_sogs
        })

    def test_normal_track_no_outliers(self):
        """Test processing of a normal track with no outliers"""
        result = _remove_outliers_row(
            self.base_row,
            threshold_partition_sog=15.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=20.0,
            threshold_association_distance=50.0,
            threshold_completeness=4
        )
        
        self.assertEqual(len(result), 1)  # Should return single processed track
        self.assertEqual(len(result[0]['timestamps']), 4)  # Should preserve all points
        self.assertTrue(isinstance(result[0]['geometry'], LineString))
        
        # Check that the track maintains the same overall shape
        coords = list(result[0]['geometry'].coords)
        self.assertEqual(len(coords), 4)
        self.assertAlmostEqual(coords[-1][1] - coords[0][1], 0.3)

    def test_track_with_speed_outlier(self):
        """Test processing of a track with a speed outlier"""
        track = LineString([
            [0, 0],
            [0, 0.1],
            [0, 1.0],  # Big jump - should be identified as outlier
            [0, 1.1]
        ])
        
        timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(minutes=61),  # Very short time after previous point
            self.base_time + timedelta(hours=3)
        ]
        
        row = pd.Series({
            'geometry': track,
            'timestamps': timestamps,
            'velocities': [10.0, 10.0, 10.0, 10.0]
        })
        
        result = _remove_outliers_row(
            row,
            threshold_partition_sog=5.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=2
        )
        
        self.assertGreater(len(result), 0)  # Should have at least one valid track
        
        # Each segment should be a valid track
        for processed_row in result:
            self.assertTrue(isinstance(processed_row['geometry'], LineString))
            self.assertEqual(len(processed_row['timestamps']), 
                           len(list(processed_row['geometry'].coords)))

    def test_track_with_distance_outlier(self):
        """Test processing of a track with a distance outlier"""
        track = LineString([
            [0, 0],
            [0, 0.1],
            [0, 5.0],  # Large distance gap (>500km)
            [0, 5.1]
        ])
        
        row = pd.Series({
            'geometry': track,
            'timestamps': self.base_timestamps,
            'velocities': self.base_sogs
        })
        
        result = _remove_outliers_row(
            row,
            threshold_partition_sog=5.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=2
        )
        
        self.assertGreater(len(result), 0)  # Should have at least one valid track
        
        # Check that each segment maintains time ordering
        for processed_row in result:
            timestamps = processed_row['timestamps']
            self.assertTrue(all(timestamps[i] < timestamps[i+1] 
                              for i in range(len(timestamps)-1)))

    def test_short_track_filtered(self):
        """Test that tracks shorter than completeness threshold are filtered out"""
        track = LineString([
            [0, 0],
            [0, 0.1]
        ])
        
        row = pd.Series({
            'geometry': track,
            'timestamps': self.base_timestamps[:2],
            'velocities': self.base_sogs[:2]
        })
        
        result = _remove_outliers_row(
            row,
            threshold_partition_sog=5.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=3
        )
        
        self.assertEqual(len(result), 0)  # Should be filtered out

    def test_track_with_multiple_segments(self):
        """Test processing of a track that should be split into multiple valid segments"""
        track = LineString([
            [0, 0], [0, 0.1],  # Segment 1
            [0, 1.0], [0, 1.1],  # Segment 2 (after gap)
            [0, 2.0], [0, 2.1]   # Segment 3 (after another gap)
        ])
        
        timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3),
            self.base_time + timedelta(hours=4),
            self.base_time + timedelta(hours=5)
        ]
        
        row = pd.Series({
            'geometry': track,
            'timestamps': timestamps,
            'velocities': [10.0] * 6
        })
        
        result = _remove_outliers_row(
            row,
            threshold_partition_sog=5.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=2
        )
        
        self.assertGreater(len(result), 0)  # Should have at least one valid track
        
        # Each segment should maintain data consistency
        for processed_row in result:
            coords = list(processed_row['geometry'].coords)
            self.assertEqual(len(coords), len(processed_row['timestamps']))
            self.assertEqual(len(coords), len(processed_row['velocities']))

    def test_data_consistency(self):
        """Test that processed tracks maintain consistency between geometry, timestamps, and velocities"""
        result = _remove_outliers_row(
            self.base_row,
            threshold_partition_sog=5.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=4
        )
        
        self.assertGreater(len(result), 0)  # Should have at least one valid track
        for processed_row in result:
            coords = list(processed_row['geometry'].coords)
            self.assertEqual(len(coords), len(processed_row['timestamps']))
            self.assertEqual(len(coords), len(processed_row['velocities']))
            
            # Check time ordering
            timestamps = processed_row['timestamps']
            self.assertTrue(all(timestamps[i] < timestamps[i+1] 
                              for i in range(len(timestamps)-1)))
            
            # Check that velocities are reasonable
            self.assertTrue(all(0 <= v <= 100 for v in processed_row['velocities']))

    def test_preserves_row_additional_fields(self):
        """Test that additional fields in the input row are preserved"""
        # Add extra fields to the base row
        row = self.base_row.copy()
        row['vessel_id'] = 'TEST123'
        row['vessel_type'] = 'cargo'
        
        result = _remove_outliers_row(
            row,
            threshold_partition_sog=5.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=4
        )
        
        self.assertGreater(len(result), 0)  # Should have at least one valid track
        for processed_row in result:
            self.assertEqual(processed_row['vessel_id'], 'TEST123')
            self.assertEqual(processed_row['vessel_type'], 'cargo')


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
            "completeness": 4
        }
        
        # Create a simple track
        self.base_track = LineString([
            [0, 0],
            [0, 0.1],
            [0, 0.2],
            [0, 0.3]
        ])
        self.base_timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3)
        ]
        self.base_sogs = [10.0, 10.0, 10.0, 10.0]

    def create_test_chunk(self, num_rows: int, include_outliers: bool = False) -> pd.DataFrame:
        """Helper function to create a test DataFrame chunk"""
        rows = []
        for i in range(num_rows):
            if include_outliers and i % 2 == 1:
                # Create track with outlier
                track = LineString([
                    [0, i],
                    [0, i + 0.1],
                    [0, i + 1.0],  # Large jump
                    [0, i + 1.1]
                ])
            else:
                # Create normal track
                track = LineString([
                    [0, i],
                    [0, i + 0.1],
                    [0, i + 0.2],
                    [0, i + 0.3]
                ])
            
            row = pd.Series({
                'geometry': track,
                'timestamps': [
                    self.base_time + timedelta(hours=j) for j in range(4)
                ],
                'velocities': [10.0] * 4,
                'vessel_id': f'VESSEL_{i}'
            })
            rows.append(row)
        
        return pd.DataFrame(rows)

    def test_process_empty_chunk(self):
        """Test processing an empty chunk"""
        empty_chunk = pd.DataFrame(columns=['geometry', 'timestamps', 'velocities'])
        args = (empty_chunk, self.thresholds, False)
        
        result = _remove_outliers_chunk(args)
        
        self.assertEqual(len(result), 0)

    def test_process_single_row_chunk(self):
        """Test processing a chunk with a single row"""
        chunk = pd.DataFrame([{
            'geometry': self.base_track,
            'timestamps': self.base_timestamps,
            'velocities': self.base_sogs,
            'vessel_id': 'VESSEL_1'
        }])
        
        args = (chunk, self.thresholds, False)
        result = _remove_outliers_chunk(args)
        
        self.assertGreater(len(result), 0)
        self.assertTrue(isinstance(result[0]['geometry'], LineString))
        self.assertEqual(result[0]['vessel_id'], 'VESSEL_1')

    def test_process_multiple_rows(self):
        """Test processing multiple rows without outliers"""
        chunk = self.create_test_chunk(5, include_outliers=False)
        args = (chunk, self.thresholds, False)
        
        result = _remove_outliers_chunk(args)
        
        self.assertEqual(len(result), 5)  # Should preserve all rows
        for row in result:
            self.assertEqual(len(list(row['geometry'].coords)), 4)
            self.assertEqual(len(row['timestamps']), 4)
            self.assertEqual(len(row['velocities']), 4)

    def test_process_rows_with_outliers(self):
        """Test processing rows containing outliers"""
        chunk = self.create_test_chunk(4, include_outliers=True)
        args = (chunk, self.thresholds, False)
        
        result = _remove_outliers_chunk(args)
        
        # Should have more rows than input due to splitting of tracks with outliers
        self.assertGreater(len(result), 0)
        
        # Check that all resulting rows are valid
        for row in result:
            coords = list(row['geometry'].coords)
            self.assertEqual(len(coords), len(row['timestamps']))
            self.assertEqual(len(coords), len(row['velocities']))

    def test_verbose_mode(self):
        """Test that verbose mode doesn't affect results"""
        chunk = self.create_test_chunk(3)
        
        # Process with verbose=False
        args_non_verbose = (chunk, self.thresholds, False)
        result_non_verbose = _remove_outliers_chunk(args_non_verbose)
        
        # Process with verbose=True
        args_verbose = (chunk, self.thresholds, True)
        result_verbose = _remove_outliers_chunk(args_verbose)
        
        # Results should be identical
        self.assertEqual(len(result_non_verbose), len(result_verbose))
        for row1, row2 in zip(result_non_verbose, result_verbose):
            self.assertTrue(row1['geometry'].equals(row2['geometry']))
            self.assertEqual(row1['timestamps'], row2['timestamps'])
            self.assertEqual(row1['velocities'], row2['velocities'])

    def test_preserve_additional_fields(self):
        """Test that additional fields are preserved in output"""
        chunk = pd.DataFrame([{
            'geometry': self.base_track,
            'timestamps': self.base_timestamps,
            'velocities': self.base_sogs,
            'vessel_id': 'VESSEL_1',
            'vessel_type': 'cargo',
            'flag': 'ABC'
        }])
        
        args = (chunk, self.thresholds, False)
        result = _remove_outliers_chunk(args)
        
        self.assertGreater(len(result), 0)
        for row in result:
            self.assertEqual(row['vessel_id'], 'VESSEL_1')
            self.assertEqual(row['vessel_type'], 'cargo')
            self.assertEqual(row['flag'], 'ABC')

    def test_row_independence(self):
        """Test that each row is processed independently"""
        # Create two identical rows
        row = pd.Series({
            'geometry': self.base_track,
            'timestamps': self.base_timestamps,
            'velocities': self.base_sogs,
            'vessel_id': 'VESSEL_1'
        })
        chunk = pd.DataFrame([row, row.copy()])
        
        args = (chunk, self.thresholds, False)
        result = _remove_outliers_chunk(args)
        
        # Results for identical rows should be identical
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0]['geometry'].equals(result[1]['geometry']))
        self.assertEqual(result[0]['timestamps'], result[1]['timestamps'])
        self.assertEqual(result[0]['velocities'], result[1]['velocities'])


class TestRemoveOutliers(TestCase):
    def setUp(self):
        """Set up common test data"""
        self.base_time = datetime(2024, 1, 1, 12, 0)
        
        # Create a simple track
        self.base_track = LineString([
            [0, 0],
            [0, 0.1],
            [0, 0.2],
            [0, 0.3]
        ])
        self.base_timestamps = [
            self.base_time,
            self.base_time + timedelta(hours=1),
            self.base_time + timedelta(hours=2),
            self.base_time + timedelta(hours=3)
        ]
        self.base_velocities = [10.0, 10.0, 10.0, 10.0]

    def create_test_df(self, num_rows: int, include_outliers: bool = False) -> pd.DataFrame:
        """Helper function to create a test DataFrame"""
        data = []
        for i in range(num_rows):
            base_lat = (i % 80)  # Keep latitude within [-90, 90]
            base_lng = (i % 80)  # Keep longitude within [-180, 180]
            
            if include_outliers and i % 2 == 1:
                # Create track with outlier, but keep coordinates valid
                track = LineString([
                    [base_lat, base_lng],
                    [base_lat, base_lng + 0.1],
                    [base_lat, base_lng + 1.0],  # Large but valid jump
                    [base_lat, base_lng + 1.1]
                ])
            else:
                # Create normal track
                track = LineString([
                    [base_lat, base_lng],
                    [base_lat, base_lng + 0.1],
                    [base_lat, base_lng + 0.2],
                    [base_lat, base_lng + 0.3]
                ])
            
            row = {
                'geometry': track,
                'timestamps': [
                    self.base_time + timedelta(hours=j) for j in range(4)
                ],
                'velocities': [10.0] * 4,
                'vessel_id': f'VESSEL_{i}',
                'vessel_type': 'cargo'
            }
            data.append(row)
        
        return pd.DataFrame(data)

    def test_empty_dataframe(self):
        """Test processing an empty DataFrame"""
        empty_df = pd.DataFrame(columns=['geometry', 'timestamps', 'velocities'])
        result = remove_outliers(empty_df)
        
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(len(result), 0)
        self.assertTrue(all(col in result.columns 
                          for col in ['geometry', 'timestamps', 'velocities']))

    def test_single_row_dataframe(self):
        """Test processing a DataFrame with a single row"""
        df = pd.DataFrame([{
            'geometry': self.base_track,
            'timestamps': self.base_timestamps,
            'velocities': self.base_velocities,
            'vessel_id': 'VESSEL_1'
        }])
        
        result = remove_outliers(df, threshold_completeness=4)
        
        self.assertEqual(len(result), 1)
        self.assertTrue(isinstance(result.iloc[0]['geometry'], LineString))
        self.assertEqual(result.iloc[0]['vessel_id'], 'VESSEL_1')
        self.assertEqual(len(result.iloc[0]['timestamps']), 4)

    def test_normal_tracks(self):
        """Test processing normal tracks without outliers"""
        df = self.create_test_df(5, include_outliers=False)
        result = remove_outliers(
            df,
            threshold_partition_sog=5.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=4
        )
        
        self.assertEqual(len(result), 5)  # Should preserve all tracks
        for _, row in result.iterrows():
            self.assertEqual(len(list(row['geometry'].coords)), 4)
            self.assertEqual(len(row['timestamps']), 4)
            self.assertEqual(len(row['velocities']), 4)

    def test_tracks_with_outliers(self):
        """Test processing tracks with outliers"""
        df = self.create_test_df(4, include_outliers=True)
        result = remove_outliers(
            df,
            threshold_partition_sog=5.0,
            threshold_partition_distance=50.0,
            threshold_association_sog=15.0,
            threshold_association_distance=50.0,
            threshold_completeness=2
        )
        
        # Number of output rows might be different due to splitting
        self.assertGreater(len(result), 0)
        
        # Check that all resulting tracks are valid
        for _, row in result.iterrows():
            coords = list(row['geometry'].coords)
            self.assertEqual(len(coords), len(row['timestamps']))
            self.assertEqual(len(coords), len(row['velocities']))

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
            threshold_completeness=2
        )
        
        # More permissive thresholds should result in fewer splits
        self.assertNotEqual(len(result_default), len(result_permissive))

    def test_preserve_dtypes(self):
        """Test that data types are preserved in the output DataFrame"""
        df = pd.DataFrame([{
            'geometry': self.base_track,
            'timestamps': self.base_timestamps,
            'velocities': self.base_velocities,
            'vessel_id': 'VESSEL_1',
            'int_field': 42,
            'float_field': 3.14,
            'bool_field': True
        }])
        
        result = remove_outliers(df, threshold_completeness=4)
        
        self.assertEqual(result['int_field'].dtype, np.int64)
        self.assertEqual(result['float_field'].dtype, np.float64)
        self.assertEqual(result['bool_field'].dtype, np.bool_)

    def test_preserve_column_order(self):
        """Test that column order is preserved in the output DataFrame"""
        columns = ['vessel_id', 'geometry', 'timestamps', 'velocities', 'extra_field']
        df = pd.DataFrame([{
            'vessel_id': 'VESSEL_1',
            'geometry': self.base_track,
            'timestamps': self.base_timestamps,
            'velocities': self.base_velocities,
            'extra_field': 'test'
        }])[columns]  # Explicitly set column order
        
        result = remove_outliers(df, threshold_completeness=4)
        
        self.assertEqual(list(result.columns), columns)

    def test_invalid_input_missing_columns(self):
        """Test handling of invalid input DataFrame missing required columns"""
        df = pd.DataFrame([{
            'vessel_id': 'VESSEL_1',
            'geometry': self.base_track,
            # Missing 'timestamps' and 'velocities'
        }])
        
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
            verbose=False
        )
        
        self.assertGreater(len(result), 0)
        # Check that all rows are valid
        for _, row in result.iterrows():
            self.assertTrue(isinstance(row['geometry'], LineString))
            self.assertEqual(len(row['timestamps']), len(list(row['geometry'].coords)))

    def test_negative_thresholds(self):
        """Test handling of negative threshold values"""
        df = self.create_test_df(1)
        
        with self.assertRaises(ValueError):
            remove_outliers(
                df,
                threshold_partition_sog=-5.0,
                threshold_partition_distance=50.0,
                threshold_association_sog=15.0,
                threshold_association_distance=50.0
            )


class TestRemoveOutliersParallel(TestCase):
    def setUp(self):
        """Set up common test data"""
        self.base_time = datetime(2024, 1, 1, 12, 0)
        self.default_thresholds = {
            'threshold_partition_sog': 15.0,
            'threshold_partition_distance': 100.0,
            'threshold_association_sog': 30.0,
            'threshold_association_distance': 100.0,
            'threshold_completeness': 4
        }
        
    def create_test_df(self, num_rows: int, include_outliers: bool = False) -> pd.DataFrame:
        """Helper function to create a test DataFrame with valid coordinates"""
        data = []
        for i in range(num_rows):
            # Use modulo to keep coordinates within valid ranges
            base_lat = (i % 80)  # Keep latitude within [-90, 90]
            base_lng = (i % 80)  # Keep longitude within [-180, 180]
            
            if include_outliers and i % 2 == 1:
                # Create track with outlier, but keep coordinates valid
                track = LineString([
                    [base_lat, base_lng],
                    [base_lat, base_lng + 0.1],
                    [base_lat, base_lng + 1.0],  # Large but valid jump
                    [base_lat, base_lng + 1.1]
                ])
            else:
                # Create normal track
                track = LineString([
                    [base_lat, base_lng],
                    [base_lat, base_lng + 0.1],
                    [base_lat, base_lng + 0.2],
                    [base_lat, base_lng + 0.3]
                ])
            
            row = {
                'geometry': track,
                'timestamps': [
                    self.base_time + timedelta(hours=j) for j in range(4)
                ],
                'velocities': [10.0] * 4,
                'vessel_id': f'VESSEL_{i}',
                'vessel_type': 'cargo'
            }
            data.append(row)
        
        return pd.DataFrame(data)

    def test_parallel_vs_single_process_results(self):
        """Test that parallel processing gives same results as single process"""
        df = self.create_test_df(10, include_outliers=True)
        
        # Process with single process
        single_result = remove_outliers(df, **self.default_thresholds, verbose=False)
        
        # Process with parallel processing
        parallel_result = remove_outliers_parallel(df, **self.default_thresholds, 
                                                 verbose=False, n_processes=2)
        
        # Compare results
        self.assertEqual(len(single_result), len(parallel_result))
        
        # Sort both DataFrames by vessel_id to ensure consistent comparison
        single_result = single_result.sort_values('vessel_id').reset_index(drop=True)
        parallel_result = parallel_result.sort_values('vessel_id').reset_index(drop=True)
        
        # Compare geometries
        for i in range(len(single_result)):
            self.assertTrue(
                single_result.iloc[i]['geometry'].equals(parallel_result.iloc[i]['geometry'])
            )
            self.assertEqual(
                single_result.iloc[i]['timestamps'], 
                parallel_result.iloc[i]['timestamps']
            )

    def test_different_process_counts(self):
        """Test processing with different numbers of processes"""
        df = self.create_test_df(20, include_outliers=True)
        
        results = {}
        process_counts = [1, 2, 4]
        
        for n_processes in process_counts:
            results[n_processes] = remove_outliers_parallel(
                df, **self.default_thresholds, verbose=False, n_processes=n_processes
            )
        
        # All results should be the same regardless of process count
        for n1 in process_counts:
            for n2 in process_counts:
                self.assertEqual(len(results[n1]), len(results[n2]))
                
                # Sort results for consistent comparison
                r1 = results[n1].sort_values('vessel_id').reset_index(drop=True)
                r2 = results[n2].sort_values('vessel_id').reset_index(drop=True)
                
                for i in range(len(r1)):
                    self.assertTrue(r1.iloc[i]['geometry'].equals(r2.iloc[i]['geometry']))

    def test_small_dataframe_single_process(self):
        """Test that small DataFrames are processed with single process"""
        df = self.create_test_df(2)
        
        start_time = time.time()
        result = remove_outliers_parallel(df, **self.default_thresholds, 
                                        verbose=False, n_processes=4)
        processing_time = time.time() - start_time
        
        self.assertGreater(len(result), 0)
        # Processing time should be consistent with single-process execution
        self.assertLess(processing_time, 1.0)  # Adjust threshold as needed

    def test_default_process_count(self):
        """Test that default process count is CPU count - 1"""
        df = self.create_test_df(10)
        
        result = remove_outliers_parallel(df, **self.default_thresholds, verbose=False)
        self.assertGreater(len(result), 0)

    def test_invalid_process_count(self):
        """Test handling of invalid process counts"""
        df = self.create_test_df(10)
        
        with self.assertRaises(ValueError):
            remove_outliers_parallel(df, **self.default_thresholds, n_processes=0)
            
        with self.assertRaises(ValueError):
            remove_outliers_parallel(df, **self.default_thresholds, n_processes=-1)

    def test_empty_dataframe(self):
        """Test processing of empty DataFrame"""
        df = pd.DataFrame(columns=['geometry', 'timestamps', 'velocities'])
        
        result = remove_outliers_parallel(df, **self.default_thresholds, 
                                        verbose=False, n_processes=2)
        
        self.assertEqual(len(result), 0)
        self.assertTrue(isinstance(result, pd.DataFrame))

    def test_preserve_dtypes(self):
        """Test that data types are preserved across parallel processing"""
        df = pd.DataFrame([{
            'geometry': LineString([[0, 0], [0, 0.1], [0, 0.2], [0, 0.3]]),
            'timestamps': [self.base_time + timedelta(hours=i) for i in range(4)],
            'velocities': [10.0] * 4,
            'vessel_id': 'VESSEL_1',
            'int_field': 42,
            'float_field': 3.14,
            'bool_field': True
        }])
        
        result = remove_outliers_parallel(df, **self.default_thresholds, 
                                        verbose=False, n_processes=2)
        
        self.assertEqual(result['int_field'].dtype, np.int64)
        self.assertEqual(result['float_field'].dtype, np.float64)
        self.assertEqual(result['bool_field'].dtype, np.bool_)

    def test_chunk_boundaries(self):
        """Test that track processing works correctly across chunk boundaries"""
        # Create DataFrame with identical consecutive tracks
        df = pd.DataFrame([{
            'geometry': LineString([[0, 0], [0, 0.1], [0, 0.2], [0, 0.3]]),
            'timestamps': [self.base_time + timedelta(hours=i) for i in range(4)],
            'velocities': [10.0] * 4,
            'vessel_id': 'VESSEL_1'
        }] * 10)  # 10 identical rows
        
        # Process with different numbers of chunks
        result_2_chunks = remove_outliers_parallel(df, **self.default_thresholds, 
                                                 verbose=False, n_processes=2)
        result_3_chunks = remove_outliers_parallel(df, **self.default_thresholds, 
                                                 verbose=False, n_processes=3)
        
        # Results should be identical regardless of chunking
        self.assertEqual(len(result_2_chunks), len(result_3_chunks))

    def test_memory_usage(self):
        """Test memory usage with large DataFrame"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large DataFrame
        df = self.create_test_df(1000, include_outliers=True)
        
        # Process in parallel
        _ = remove_outliers_parallel(df, **self.default_thresholds, 
                                   verbose=False, n_processes=4)
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (adjust threshold as needed)
        self.assertLess(memory_increase, 1000)  # Less than 1GB increase
