# Walmart weekly sales study

This project analyzes the supplied weekly sales for 45 stores and compares one-week-ahead forecasts. The analysis uses historical sales, the holiday flag, and prior-week temperature, fuel price, consumer price index, and unemployment. It treats these fields as predictive associations rather than causal effects.

## Environment

The project environment is `.venv`, using Python 3.12. NumPy, pandas, scikit-learn, matplotlib, and their dependencies are installed there. From PowerShell in this folder, run:

```powershell
.\.venv\Scripts\python.exe run.py
```

To install the declared dependencies in a fresh environment, use:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The command reads `Walmart_Sales.csv` and writes its results to `reports`. You can choose other paths:

```powershell
.\.venv\Scripts\python.exe run.py --data .\Walmart_Sales.csv --output .\reports
```

## What the run produces

- `reports/analysis.md`: method, model comparison, held-out results, and interpretation.
- `reports/validation_metrics.csv` and `validation_predictions.csv`: four chronological validation folds.
- `reports/holdout_metrics.csv`, `holdout_predictions.csv`, and `store_holdout_metrics.csv`: the final 13 weeks.
- `reports/store_summary.csv`, `holiday_summary.csv`, and `factor_correlations.csv`: descriptive analysis.
- `reports/factor_ablation.csv`: effect of adding prior-week weather and economic fields when a trained model is selected.
- `reports/weekly_sales.png` and `holdout_forecast.png`: sales history and held-out forecast chart.

The model is chosen using validation WAPE. The held-out period is used only after selection. Each historical prediction assumes the prior week's actual sales are available. The first 52 weeks per store provide sales history for lag features.

The data has no discount, stock, margin, or operating-cost columns. The project therefore cannot quantify discount effects or financial savings. Future operational forecasts also need a known holiday calendar and the previous week's observed inputs.

The project does not automatically run when installed. Execute the command above when you are ready to inspect and verify the results.
