import pandas as pd
from shapely.geometry import Polygon


def filter_data(data: pd.DataFrame, area_of_interest: Polygon, only_inside: bool = True):
    """
    Get all trajectories in the area defined by the borders.

    Args:
        data: Dataframe with column "geometry" containing the trajectories in the form of shapely LineStrings.
        area_of_interest: Polygon defining the area of interest.
        only_inside: If True, only trajectories that are completely inside the area of interest are returned.

    Returns:
        Dataframe with the filtered trajectories.
    """
    if "geometry" not in data.columns:
        raise ValueError("Dataframe must contain a column 'geometry' with the trajectories")

    if only_inside:
        data = data[data.apply(lambda x: area_of_interest.contains(x["geometry"]), axis=1)]
    else:
        data = data[data.apply(lambda x: area_of_interest.intersects(x["geometry"]), axis=1)]
    return data
