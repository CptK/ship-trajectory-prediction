import ssl

import geopandas as gpd
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context


COUNTRIES_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
ALL_COUNTRIES = gpd.read_file(COUNTRIES_URL)


def plot_world(ax: plt.Axes) -> None:
    ALL_COUNTRIES.plot(ax=ax, color="white", edgecolor="gray")


def plot_north_america(ax: plt.Axes) -> None:
    north_america = ALL_COUNTRIES[ALL_COUNTRIES["CONTINENT"] == "North America"]
    north_america.plot(ax=ax, color="white", edgecolor="gray")


def plot_europe(ax: plt.Axes, main_land: bool = True) -> None:
    europe = ALL_COUNTRIES[ALL_COUNTRIES["CONTINENT"] == "Europe"]
    europe.plot(ax=ax, color="white", edgecolor="gray")
    # set axis limits so only Europe is shown
    if main_land:
        ax.set_xlim(-10, 40)
        ax.set_ylim(35, 70)
