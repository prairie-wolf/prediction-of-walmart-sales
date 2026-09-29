"""Run the complete, reproducible analysis."""

from pathlib import Path

import pandas as pd

from .analysis import descriptive_tables, save_charts
from .data import BASE_FEATURES, FULL_FEATURES, load_sales, make_features
from .models import MODEL_NAMES, by_store_metrics, cross_validate, metrics, predict


def _report(
    source: pd.DataFrame,
    training: pd.DataFrame,
    holdout: pd.DataFrame,
    validation_scores: pd.DataFrame,
    holdout_scores: pd.DataFrame,
    ablation_scores: pd.DataFrame,
) -> str:
    selected = validation_scores.iloc[0]["model"]
    selected_test = holdout_scores.loc[holdout_scores["model"] == selected].iloc[0]
    holiday_rows = holdout.groupby("Holiday_Flag")
    holiday_lines = [
        f"- Holiday flag {int(flag)}: {len(group)} store-weeks, WAPE {metrics(group['Weekly_Sales'], group['prediction'])['wape']:.1%}"
        for flag, group in holiday_rows
    ]
    validation_lines = [
        f"| {row.model} | {row.mae:,.0f} | {row.rmse:,.0f} | {row.wape:.1%} |"
        for row in validation_scores.itertuples(index=False)
    ]
    test_lines = [
        f"| {row.model} | {row.mae:,.0f} | {row.rmse:,.0f} | {row.wape:.1%} |"
        for row in holdout_scores.itertuples(index=False)
    ]
    factor_text = (
        "Ablation was skipped because the selected model uses only sales history."
        if ablation_scores.empty
        else "\n".join(
            f"- {row.feature_set}: validation WAPE {row.wape:.1%}"
            for row in ablation_scores.itertuples(index=False)
        )
    )
    return f"""# Walmart weekly sales study

## Data and method

The source has {len(source):,} store-weeks across {source['Store'].nunique()} stores, from {source['Date'].min():%Y-%m-%d} to {source['Date'].max():%Y-%m-%d}. One row is a store's weekly sales. The supplied temperature and fuel price are treated as source units; the file does not state them explicitly.

Models use the holiday flag for the forecast week, past sales, and the previous week's weather and economic measures. The first 52 weeks of each store supply lagged history. The final {holdout['Date'].nunique()} weeks are reserved for testing. Earlier data is scored in four expanding-window validation folds of eight weeks each. Every store in a week stays in the same fold. Each prediction assumes the previous week's actual sales are available, so the study measures rolling one-week-ahead forecasts.

## Model comparison

Validation results, pooled across folds:

| Model | MAE per store-week | RMSE per store-week | WAPE |
|---|---:|---:|---:|
{chr(10).join(validation_lines)}

Selected by lowest validation WAPE: **{selected}**. The held-out WAPE for this model is **{selected_test['wape']:.1%}**.

Held-out results:

| Model | MAE per store-week | RMSE per store-week | WAPE |
|---|---:|---:|---:|
{chr(10).join(test_lines)}

Selected model, by holiday flag in the held-out period:

{chr(10).join(holiday_lines)}

## Factors and interpretation

`factor_correlations.csv` measures correlations after removing each store's average. `holiday_summary.csv` is a raw group comparison. Both are descriptive, and seasonality or broader trends may account for part of the relationships.

The model ablation compares validation error with and without prior-week temperature, fuel price, CPI, and unemployment:

{factor_text}

The ablation measures whether these fields improve prediction in this dataset. It does not measure causal effects. The file contains no discount, inventory, margin, or operating-cost data, so direct discount effects and monetary savings cannot be estimated from it. Any cost or economic impact would require separate operational and financial inputs.

## Outputs

The CSV files contain the numerical results, including each held-out store-week forecast and store-level errors. `weekly_sales.png` and `holdout_forecast.png` show the sales history and held-out total forecasts.
"""


def run(source_path: str | Path, output_dir: str | Path) -> Path:
    source = load_sales(source_path)
    featured = make_features(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    test_dates = pd.Index(sorted(featured["Date"].unique())[-13:])
    training = featured.loc[~featured["Date"].isin(test_dates)].copy()
    test = featured.loc[featured["Date"].isin(test_dates)].copy()
    validation_scores, validation_predictions = cross_validate(training)
    selected = str(validation_scores.iloc[0]["model"])

    holdout_rows = []
    holdout_scores = []
    for name in MODEL_NAMES:
        result = test[["Store", "Date", "Weekly_Sales", "Holiday_Flag"]].copy()
        result["prediction"] = predict(name, training, test)
        result["model"] = name
        holdout_rows.append(result)
        holdout_scores.append({"model": name, **metrics(result["Weekly_Sales"], result["prediction"])})
    all_holdout = pd.concat(holdout_rows, ignore_index=True)
    test_scores = pd.DataFrame(holdout_scores).sort_values("wape")
    selected_predictions = all_holdout.loc[all_holdout["model"] == selected].copy()

    ablation = pd.DataFrame()
    if selected in ("ridge", "gradient_boosting"):
        reduced_scores, _ = cross_validate(training, names=[selected], features=BASE_FEATURES)
        ablation = pd.DataFrame(
            [
                {"feature_set": "sales and calendar", "wape": reduced_scores.iloc[0]["wape"]},
                {"feature_set": "sales, calendar, and prior-week factors", "wape": validation_scores.loc[validation_scores["model"] == selected, "wape"].iloc[0]},
            ]
        )

    for name, table in descriptive_tables(source).items():
        table.to_csv(output / f"{name}.csv", index=False)
    validation_scores.to_csv(output / "validation_metrics.csv", index=False)
    validation_predictions.to_csv(output / "validation_predictions.csv", index=False)
    test_scores.to_csv(output / "holdout_metrics.csv", index=False)
    selected_predictions.to_csv(output / "holdout_predictions.csv", index=False)
    by_store_metrics(selected_predictions).to_csv(output / "store_holdout_metrics.csv", index=False)
    if not ablation.empty:
        ablation.to_csv(output / "factor_ablation.csv", index=False)
    save_charts(source, selected_predictions, output)
    (output / "analysis.md").write_text(
        _report(source, training, selected_predictions, validation_scores, test_scores, ablation),
        encoding="utf-8",
    )
    return output
