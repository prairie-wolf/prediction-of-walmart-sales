"""Descriptive statistics and charts for the source and holdout period."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import DRIVERS


def descriptive_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    store_summary = (
        frame.groupby("Store", as_index=False)
        .agg(weeks=("Date", "size"), average_weekly_sales=("Weekly_Sales", "mean"), total_sales=("Weekly_Sales", "sum"))
        .sort_values("Store")
    )
    holiday_summary = (
        frame.groupby("Holiday_Flag", as_index=False)
        .agg(weeks=("Weekly_Sales", "size"), average_weekly_sales=("Weekly_Sales", "mean"))
    )
    # Remove each store's average before measuring within-store association.
    columns = ["Weekly_Sales", *DRIVERS]
    centered = frame[columns] - frame.groupby("Store")[columns].transform("mean")
    correlations = pd.DataFrame(
        {"factor": DRIVERS, "within_store_correlation": [centered["Weekly_Sales"].corr(centered[d]) for d in DRIVERS]}
    )
    return {
        "store_summary": store_summary,
        "holiday_summary": holiday_summary,
        "factor_correlations": correlations,
    }


def save_charts(frame: pd.DataFrame, holdout: pd.DataFrame, output: Path) -> None:
    weekly = frame.groupby("Date", as_index=False)["Weekly_Sales"].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(weekly["Date"], weekly["Weekly_Sales"] / 1e6, linewidth=1.7)
    ax.set(title="Total weekly sales across 45 stores", ylabel="Sales (millions)", xlabel="Week")
    ax.grid(axis="y", alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "weekly_sales.png", dpi=160)
    plt.close(fig)

    observed = holdout.groupby("Date", as_index=False)[["Weekly_Sales", "prediction"]].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(observed["Date"], observed["Weekly_Sales"] / 1e6, marker="o", label="Actual")
    ax.plot(observed["Date"], observed["prediction"] / 1e6, marker="o", label="Forecast")
    ax.set(title="Held-out weekly sales", ylabel="Sales (millions)", xlabel="Week")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "holdout_forecast.png", dpi=160)
    plt.close(fig)
