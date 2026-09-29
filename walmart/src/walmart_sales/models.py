"""Chronological validation and comparable forecasting models."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FULL_FEATURES


MODEL_NAMES = ("last_week", "last_year", "ridge", "gradient_boosting")


def metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    observed = np.asarray(actual, dtype=float)
    forecast = np.asarray(predicted, dtype=float)
    errors = observed - forecast
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "wape": float(np.sum(np.abs(errors)) / np.sum(np.abs(observed))),
    }


def make_model(name: str, features: Sequence[str] = FULL_FEATURES) -> Pipeline:
    numeric = [feature for feature in features if feature != "Store"]
    preprocessor = ColumnTransformer(
        [
            ("store", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Store"]),
            ("numeric", StandardScaler(), numeric),
        ],
        sparse_threshold=0,
    )
    if name == "ridge":
        estimator = Ridge(alpha=100.0)
    elif name == "gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_iter=250,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown trained model: {name}")
    return Pipeline([("features", preprocessor), ("model", estimator)])


def predict(
    name: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: Sequence[str] = FULL_FEATURES,
) -> np.ndarray:
    if name == "last_week":
        return evaluation["sales_lag1"].to_numpy()
    if name == "last_year":
        return evaluation["sales_lag52"].to_numpy()
    model = make_model(name, features)
    model.fit(train[list(features)], train["Weekly_Sales"])
    return model.predict(evaluation[list(features)])


def validation_windows(
    dates: Sequence[pd.Timestamp], horizon: int = 8, folds: int = 4
) -> list[tuple[pd.Index, pd.Index]]:
    """Use expanding training windows and the next eight weeks for each fold."""
    ordered = pd.Index(sorted(pd.unique(dates)))
    initial = len(ordered) - horizon * folds
    if initial < 30:
        raise ValueError("At least 30 training weeks are needed before validation.")
    return [
        (ordered[: initial + i * horizon], ordered[initial + i * horizon : initial + (i + 1) * horizon])
        for i in range(folds)
    ]


def cross_validate(
    frame: pd.DataFrame,
    names: Sequence[str] = MODEL_NAMES,
    features: Sequence[str] = FULL_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool out-of-fold predictions for fair model ranking."""
    rows: list[pd.DataFrame] = []
    for fold_number, (train_dates, validation_dates) in enumerate(
        validation_windows(frame["Date"]), start=1
    ):
        train = frame.loc[frame["Date"].isin(train_dates)]
        validation = frame.loc[frame["Date"].isin(validation_dates)]
        for name in names:
            result = validation[["Store", "Date", "Weekly_Sales", "Holiday_Flag"]].copy()
            result["prediction"] = predict(name, train, validation, features)
            result["model"] = name
            result["fold"] = fold_number
            rows.append(result)
    predictions = pd.concat(rows, ignore_index=True)
    scores = []
    for name, group in predictions.groupby("model", sort=False):
        scores.append({"model": name, **metrics(group["Weekly_Sales"], group["prediction"])})
    return pd.DataFrame(scores).sort_values("wape").reset_index(drop=True), predictions


def by_store_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for store, group in predictions.groupby("Store", sort=True):
        rows.append({"Store": store, **metrics(group["Weekly_Sales"], group["prediction"])})
    return pd.DataFrame(rows).sort_values("wape", ascending=False).reset_index(drop=True)
