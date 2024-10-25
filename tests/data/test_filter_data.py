from unittest import TestCase

import pandas as pd
from shapely.geometry import LineString, Polygon

from prediction.data import filter_data


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
