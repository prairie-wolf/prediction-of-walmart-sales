# Walmart weekly sales study

## Data and method

The source has 6,435 store-weeks across 45 stores, from 2010-02-05 to 2012-10-26. One row is a store's weekly sales. The supplied temperature and fuel price are treated as source units; the file does not state them explicitly.

Models use the holiday flag for the forecast week, past sales, and the previous week's weather and economic measures. The first 52 weeks of each store supply lagged history. The final 13 weeks are reserved for testing. Earlier data is scored in four expanding-window validation folds of eight weeks each. Every store in a week stays in the same fold. Each prediction assumes the previous week's actual sales are available, so the study measures rolling one-week-ahead forecasts.

## Model comparison

Validation results, pooled across folds:

| Model | MAE per store-week | RMSE per store-week | WAPE |
|---|---:|---:|---:|
| last_year | 61,992 | 90,293 | 5.9% |
| gradient_boosting | 78,839 | 123,010 | 7.5% |
| ridge | 84,322 | 130,235 | 8.0% |
| last_week | 94,040 | 192,054 | 8.9% |

Selected by lowest validation WAPE: **last_year**. The held-out WAPE for this model is **5.1%**.

Held-out results:

| Model | MAE per store-week | RMSE per store-week | WAPE |
|---|---:|---:|---:|
| gradient_boosting | 36,653 | 56,147 | 3.6% |
| ridge | 42,097 | 58,011 | 4.1% |
| last_week | 49,266 | 74,549 | 4.8% |
| last_year | 52,740 | 84,617 | 5.1% |

Selected model, by holiday flag in the held-out period:

- Holiday flag 0: 540 store-weeks, WAPE 5.1%
- Holiday flag 1: 45 store-weeks, WAPE 6.0%

## Factors and interpretation

`factor_correlations.csv` measures correlations after removing each store's average. `holiday_summary.csv` is a raw group comparison. Both are descriptive, and seasonality or broader trends may account for part of the relationships.

The model ablation compares validation error with and without prior-week temperature, fuel price, CPI, and unemployment:

Ablation was skipped because the selected model uses only sales history.

The ablation measures whether these fields improve prediction in this dataset. It does not measure causal effects. The file contains no discount, inventory, margin, or operating-cost data, so direct discount effects and monetary savings cannot be estimated from it. Any cost or economic impact would require separate operational and financial inputs.

## Outputs

The CSV files contain the numerical results, including each held-out store-week forecast and store-level errors. `weekly_sales.png` and `holdout_forecast.png` show the sales history and held-out total forecasts.
