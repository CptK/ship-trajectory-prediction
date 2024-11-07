from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data"


def vessel_groups() -> dict:
    """Return a dictionary of vessel types and their corresponding descriptions."""
    vessel_types_path = DATA_PATH / "AIS_categories.csv"
    types_df = pd.read_csv(vessel_types_path)
    categories = {
        r["num"]: r["category"] if r["category"] in [0, 2, 3, 19, 12, 18] else 21
        for i, r in types_df.iterrows()
    }
    categories[np.nan] = 0

    def category_desc(val):
        """Return description for the category with the indicated integer value"""
        return types_df[types_df["category"] == val].iloc[0]["category_desc"]

    groups = {categories[i]: category_desc(categories[i]) for i in types_df["num"].unique()}
    return groups


def status_codes() -> dict:
    """Return a dictionary of status codes and their corresponding descriptions."""
    status_codes_path = DATA_PATH / "AIS_status_codes.csv"
    codes_df = pd.read_csv(status_codes_path)
    return {r["code"]: r["description"] for _, r in codes_df.iterrows()}
