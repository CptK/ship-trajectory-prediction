import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def _build_url(year: int, month: int, day: int) -> str:
    """Build the data download url for the given year, month, and day"""
    m = str(month).zfill(2)
    d = str(day).zfill(2)
    return f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{m}_{d}.zip"


def _build_filename(year: int, month: int, day: int) -> str:
    """Build the filename for the given year, month, and day"""
    return f"AIS_{year}_{str(month).zfill(2)}_{str(day).zfill(2)}.csv"


def download_data(date: datetime, override: bool = False, verbose: bool = True) -> None:
    """Download the data for the given date.
    
    The results are saved in the data/raw directory. If the file already exists, it will not be downloaded
    again unless the override flag is set to True."""
    year, month, day = date.year, date.month, date.day
    save_path = Path(__file__).resolve().parents[2] / "data" / "raw"

    # check if the file already exists
    if (save_path / _build_filename(year, month, day)).exists() and not override:
        if verbose:
            print(f"Data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)} already exists in {save_path}")
        return

    response = requests.get(_build_url(year, month, day))

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(save_path)

    if verbose:
        print(f"Successfully downloaded data for {year}-{str(month).zfill(2)}-{str(day).zfill(2)} into {save_path}")


def download_all_data(start_date: datetime, end_date: datetime) -> None:
    """Download all data between the start_date and end_date, both inclusive."""
    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    for date in tqdm(all_dates):
        download_data(date, verbose=False)


def load_data(start_date: datetime, end_date: datetime | None = None) -> pd.DataFrame:
    """Load the data between the start_date and end_date, both inclusive.
    
    If no end date is specified, only the data for the start_date is loaded."""
    if end_date is None:
        end_date = start_date

    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    dfs = []
    for date in tqdm(all_dates):
        year, month, day = date.year, date.month, date.day
        save_path = Path(__file__).resolve().parents[2] / "data" / "raw"
        filename = _build_filename(year, month, day)
        if (save_path / filename).exists():
            df = pd.read_csv(save_path / filename, delimiter=",", encoding='utf-8', on_bad_lines="skip", header=0, parse_dates=["BaseDateTime"])
            dfs.append(df)
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.sort_values(['MMSI', 'BaseDateTime'])
    df["VesselType"] = df["VesselType"].astype('Int64')
    return final_df

# download_data(datetime(2024, 1, 1))
# download_all_data(datetime(2024, 1, 1), datetime(2024, 1, 10))
