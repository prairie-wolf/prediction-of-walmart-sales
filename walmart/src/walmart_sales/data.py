"""Load the source data and create features available at forecast time."""

from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = [
    "Store", "Date", "Weekly_Sales", "Holiday_Flag", "Temperature",
    "Fuel_Price", "CPI", "Unemployment",
]
DRIVERS = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
LAGGED_DRIVERS = [f"{name}_lag1" for name in DRIVERS]
BASE_FEATURES = [
    "Store", "Holiday_Flag", "week_sin", "week_cos", "year_index",
    "sales_lag1", "sales_lag4", "sales_lag52", "sales_mean4",
]
FULL_FEATURES = BASE_FEATURES + LAGGED_DRIVERS


def load_sales(path: str | Path) -> pd.DataFrame:
    """Read the original CSV without guessing its day-month-year date format."""
    frame = pd.read_csv(path)
    if list(frame.columns) != COLUMNS:
        raise ValueError(f"Expected columns in this order: {COLUMNS}")
    frame["Date"] = pd.to_datetime(frame["Date"], format="%d-%m-%Y", errors="raise")
    for column in COLUMNS:
        if column != "Date":
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame[COLUMNS].isna().any().any():
        raise ValueError("The source contains missing values.")
    if frame.duplicated(["Store", "Date"]).any():
        raise ValueError("Store and Date must identify a single row.")
    if not frame["Holiday_Flag"].isin([0, 1]).all():
        raise ValueError("Holiday_Flag must be 0 or 1.")
    if (frame["Weekly_Sales"] < 0).any():
        raise ValueError("Weekly_Sales cannot be negative.")
    frame = frame.sort_values(["Store", "Date"]).reset_index(drop=True)
    gaps = frame.groupby("Store")["Date"].diff().dropna()
    if not gaps.eq(pd.Timedelta(days=7)).all():
        raise ValueError("Each store must have consecutive seven-day observations.")
    return frame


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one-week-ahead features using current calendar and past observations."""
    result = frame.sort_values(["Store", "Date"]).copy()
    by_store = result.groupby("Store", sort=False)
    result["Store"] = result["Store"].astype(str)
    iso_week = result["Date"].dt.isocalendar().week.astype(int)
    result["week_sin"] = np.sin(2 * np.pi * iso_week / 52.1775)
    result["week_cos"] = np.cos(2 * np.pi * iso_week / 52.1775)
    result["year_index"] = (result["Date"] - result["Date"].min()).dt.days / 365.25
    for lag in (1, 4, 52):
        result[f"sales_lag{lag}"] = by_store["Weekly_Sales"].shift(lag)
    result["sales_mean4"] = by_store["Weekly_Sales"].transform(
        lambda values: values.shift(1).rolling(4).mean()
    )
    for driver in DRIVERS:
        result[f"{driver}_lag1"] = by_store[driver].shift(1)
    return result.dropna(subset=FULL_FEATURES).reset_index(drop=True)
