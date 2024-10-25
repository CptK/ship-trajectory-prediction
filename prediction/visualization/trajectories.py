import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def plot_trajectories(df: pd.DataFrame | gpd.GeoDataFrame, ax: plt.Axes) -> None:
    """Plot the trajectories in the given DataFrame on the given axis.

    The DataFrame should have a 'geometry' column containing the trajectory points and a 'vessel_type' column
    containing the vessel type for each trajectory.

    Args:
        df: The DataFrame containing the trajectories and vessel types.
        ax: The axis to plot the trajectories on.
    """
    if not isinstance(df, gpd.GeoDataFrame):
        df = gpd.GeoDataFrame(df, geometry="geometry")

    df.plot(ax=ax, linewidth=1, categorical=True, cmap="viridis", column="vessel_type", legend=True)
