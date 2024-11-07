from multiprocessing import cpu_count

import dask.dataframe as dd
import pandas as pd
from shapely import wkt
from shapely.geometry import Polygon

from prediction.preprocessing.utils import pairwise_point_distances


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


def filter_by_travelled_distance(
    data: pd.DataFrame,
    min_dist: float | None,
    max_dist: float | None,
    dist_col: str | None = None,
    n_processes: int | None = None,
):
    """Filter trajectories by the distance travelled.

    Remove trajectories that have travelled less than min_dist or more than max_dist. If no dist_col is
    provided, the function will calculate the distance from the trajectory geometry using the haversine
    formula.

    Args:
        data: Dataframe with the trajectories.
        min_dist: Minimum distance travelled in meters.
        max_dist: Maximum distance travelled in meters.
        dist_col: Column name containing the distance travelled. If None, the distance will be calculated.
        n_processes: Number of processes to use for parallel computation. If None, all available CPUs are used

    Returns:
        Dataframe with the filtered trajectories.

    Raises:
        ValueError: If dist_col is not None and not in the dataframe.
        ValueError: If dist_col is not None and df does not contain a "geometry" column.
        ValueError: If min_dist and max_dist are both None.
    """
    if dist_col is not None and dist_col not in data.columns:
        raise ValueError(f"Column {dist_col} not found in dataframe")

    if dist_col is not None and "geometry" not in data.columns:
        raise ValueError("Dataframe must contain a 'geometry' column if dist_col is not None")

    if min_dist is None and max_dist is None:
        raise ValueError("At least one of min_dist or max_dist must be provided")

    if dist_col is None:
        n_processes = n_processes or cpu_count()
        # Create a copy to avoid modifying the original
        data = data.copy()

        # Convert geometry objects to WKT before creating dask dataframe
        data["geometry_wkt"] = data["geometry"].apply(lambda x: x.wkt)

        # Convert to dask dataframe
        ddf = dd.from_pandas(data, npartitions=n_processes)

        # Calculate distances using WKT strings
        data["max_dist"] = (
            ddf["geometry_wkt"]
            .map(lambda x: pairwise_point_distances(wkt.loads(x)).max(), meta=("max_dist", "float64"))
            .compute()
        )

        # Clean up temporary column
        data = data.drop(columns=["geometry_wkt"])
        dist_col = "max_dist"

    if min_dist is not None:
        data = data[data[dist_col] >= min_dist]

    if max_dist is not None:
        data = data[data[dist_col] <= max_dist]

    return data
