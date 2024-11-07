from unittest import TestCase

import pandas as pd
from shapely.geometry import LineString, Polygon

from prediction.data.filter_data import filter_by_travelled_distance, filter_data


class TestFilterData(TestCase):
    def setUp(self):
        self.str1 = LineString([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
        self.str2 = LineString([(2, 2), (3, 3), (4, 4)])
        self.str3 = LineString([(4, 4), (5, 5)])
        self.str4 = LineString([(-0.5, -0.5), (-0.5, 1), (0, 0), (1, 1), (2, 2)])
        self.str5 = LineString([(1, 4), (2, 7), (3, 5)])
        self.str6 = LineString([(-2, -2), (-1, -2), (0, -2)])
        self.str7 = LineString([(0, 0), (1, 1), (2, 2)])

        self.data = pd.DataFrame(
            {"geometry": [self.str1, self.str2, self.str3, self.str4, self.str5, self.str6, self.str7]}
        )

        self.area_of_interest = Polygon([(-1, -1), (-1, 3), (3, 3), (3, -1)])

    def test_filter_data_inside(self):
        result = filter_data(self.data, self.area_of_interest, only_inside=True).reset_index(drop=True)
        self.assertEqual(len(result), 2)
        expected = pd.DataFrame({"geometry": [self.str4, self.str7]})
        pd.testing.assert_frame_equal(result, expected)

    def test_filter_data_not_only_inside(self):
        result = filter_data(self.data, self.area_of_interest, only_inside=False).reset_index(drop=True)
        self.assertEqual(len(result), 4)
        expected = pd.DataFrame({"geometry": [self.str1, self.str2, self.str4, self.str7]})
        pd.testing.assert_frame_equal(result, expected)


class TestFilterByTravelledDistance(TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create sample trajectories
        trajectories = [
            LineString([(0, 0), (1, 1)]),  # ~157km
            LineString([(0, 0), (2, 2)]),  # ~314km
            LineString([(0, 0), (3, 3)]),  # ~471km
            LineString([(0, 0), (4, 4)]),  # ~628km
        ]

        # Create GeoDataFrame with trajectories
        self.gdf = pd.DataFrame(
            {
                "geometry": trajectories,
                "id": range(len(trajectories)),
                "known_distance": [157, 314, 471, 628],  # distances in meters
            }
        )

    def test_filter_with_min_dist(self):
        """Test filtering with only minimum distance."""
        result = filter_by_travelled_distance(
            self.gdf, min_dist=300, max_dist=None, dist_col="known_distance"
        )
        self.assertEqual(len(result), 3)  # Should keep trajectories 2, 3, and 4
        self.assertTrue(all(id in result["id"].values for id in [1, 2, 3]))

    def test_filter_with_max_dist(self):
        """Test filtering with only maximum distance."""
        result = filter_by_travelled_distance(
            self.gdf, min_dist=None, max_dist=500, dist_col="known_distance"
        )
        self.assertEqual(len(result), 3)  # Should keep trajectories 1, 2, and 3
        self.assertTrue(all(id in result["id"].values for id in [0, 1, 2]))

    def test_filter_with_both_limits(self):
        """Test filtering with both minimum and maximum distance."""
        result = filter_by_travelled_distance(self.gdf, min_dist=200, max_dist=500, dist_col="known_distance")
        self.assertEqual(len(result), 2)  # Should keep trajectories 2 and 3
        self.assertTrue(all(id in result["id"].values for id in [1, 2]))

    def test_all_trajectories_kept(self):
        """Test when all trajectories meet the criteria."""
        result = filter_by_travelled_distance(
            self.gdf, min_dist=100, max_dist=1000, dist_col="known_distance"
        )
        self.assertEqual(len(result), len(self.gdf))
        self.assertTrue(all(id in result["id"].values for id in self.gdf["id"].values))

    def test_invalid_distance_column(self):
        """Test error raised when invalid distance column provided."""
        with self.assertRaises(ValueError):
            filter_by_travelled_distance(self.gdf, min_dist=0, max_dist=1000, dist_col="nonexistent_column")

    def test_no_geometry_column(self):
        """Test error raised when no geometry column present."""
        df_no_geometry = self.gdf.drop(columns=["geometry"])
        with self.assertRaises(ValueError):
            filter_by_travelled_distance(df_no_geometry, min_dist=0, max_dist=1000, dist_col="known_distance")

    def test_no_distance_limits(self):
        """Test error raised when neither min_dist nor max_dist provided."""
        with self.assertRaises(ValueError):
            filter_by_travelled_distance(self.gdf, min_dist=None, max_dist=None)

    def test_zero_distance_filter(self):
        """Test filtering with zero minimum distance."""
        result = filter_by_travelled_distance(self.gdf, min_dist=0, max_dist=1000, dist_col="known_distance")
        self.assertEqual(len(result), len(self.gdf))

    def test_exact_boundary_values(self):
        """Test filtering with exact boundary values."""
        # Test with exact known distance value
        result = filter_by_travelled_distance(self.gdf, min_dist=314, max_dist=314, dist_col="known_distance")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["id"], 1)

    def test_float_integer_compatibility(self):
        """Test compatibility between float and integer distance values."""
        # Test with integer values
        result_int = filter_by_travelled_distance(
            self.gdf, min_dist=300, max_dist=500, dist_col="known_distance"
        )
        # Test with float values
        result_float = filter_by_travelled_distance(
            self.gdf, min_dist=300.0, max_dist=500.0, dist_col="known_distance"
        )
        # Results should be identical
        pd.testing.assert_frame_equal(result_int, result_float)

    def test_no_distance_column(self):
        """Test filtering when no distance column provided."""
        result = filter_by_travelled_distance(self.gdf, min_dist=0, max_dist=None)
        result["max_dist"] = result["max_dist"].astype(int)
        self.assertTrue((result["max_dist"].values == self.gdf["known_distance"].values).all())

        result = filter_by_travelled_distance(self.gdf, min_dist=None, max_dist=500)
        result = result.copy()
        result["max_dist"] = result["max_dist"].astype(int)
        self.assertTrue((result["max_dist"].values == self.gdf["known_distance"].values[:3]).all())
