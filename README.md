# 🛍️ Retail Sales Pipeline & ML Forecasting Engine — v2.0

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-v2.0%2B-darkblue.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A production-grade, configuration-driven data engineering and machine learning pipeline. This project simulates a messy transaction dataset, executes a rigorous 10-step cleaning workflow, performs advanced statistical analytics, trains a Random Forest sales forecasting model, and outputs visual dashboards alongside executive business reports.

---

## 📁 System Architecture

```text
                                  ┌────────────────────────┐
                                  │   config/config.yaml   │ (Central Configuration)
                                  └───────────┬────────────┘
                                              │
                                              ▼
 ┌────────────────────┐          ┌────────────────────────┐          ┌────────────────────┐
 │  generate_data.py  │ ───────► │   raw_sales_data.csv   │ ───────► │  data_cleaner.py   │
 └────────────────────┘          └────────────────────────┘          └─────────┬──────────┘
                                                                               │
                                                                               ▼
 ┌────────────────────┐          ┌────────────────────────┐          ┌────────────────────┐
 │  ml_forecaster.py  │ ◄─────── │  cleaned_sales_data.csv│ ◄─────── │   10-Step Clean    │
 └─────────┬──────────┘          └───────────┬────────────┘          └────────────────────┘
           │                                 │
           ▼ (Forecasts)                     ▼ (Analytics)
 ┌────────────────────┐          ┌────────────────────────┐          ┌────────────────────┐
 │ sales_forecast.pkl │          │      analyzer.py       │ ───────► │   visualizer.py    │
 └─────────┬──────────┘          └───────────┬────────────┘          └─────────┬──────────┘
           │                                 │                                 │
           └────────────────► ┌──────────────▼──────────────┐ ◄────────────────┘
                              │    report_generator.py      │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────┐
                              │ outputs/reports/            │
                              │  ├── data_quality_report.txt│
                              │  └── executive_summary.txt  │
                              └─────────────────────────────┘
```

---

## 📂 Project Structure

```directory
1st/
├── config/
│   └── config.yaml                ← Centralised configuration parameters
├── data/
│   ├── raw_sales_data.csv         ← Auto-generated messy transaction logs
│   └── cleaned_sales_data.csv     ← Production-ready cleaned dataset
├── outputs/
│   ├── dashboards/                ← 5 dark-mode publication dashboards
│   │   ├── dashboard_1_overview.png
│   │   ├── dashboard_2_distributions.png
│   │   ├── dashboard_3_correlations.png
│   │   ├── dashboard_4_trends.png
│   │   └── dashboard_5_ml_insights.png
│   ├── models/
│   │   └── sales_forecast.pkl            ← Saved Random Forest model
│   ├── reports/
│   │   ├── data_quality_report.txt       ← Detailed cleaning metrics & DQ score
│   │   ├── executive_summary.txt         ← Business KPIs & ML forecasts
│   │   └── statistical_analysis.txt      ← ANOVA, Chi-Square, & normality results
│   └── logs/
│       └── pipeline.log                  ← Persistent execution logs
├── src/
│   ├── logger.py                  ← Centralised logging factory
│   ├── generate_data.py           ← Messy dataset generator (3,000+ rows)
│   ├── data_cleaner.py            ← 10-step cleaning pipeline
│   ├── analyzer.py                ← Advanced statistical analysis module
│   ├── ml_forecaster.py           ← RF forecasting with Time-Series CV
│   ├── visualizer.py              ← Matplotlib dashboard plotting engine
│   ├── report_generator.py        ← Stakeholder report writer
│   └── main.py                    ← Pipeline orchestrator
├── tests/
│   └── test_cleaner.py            ← Unit test suite for cleaner stages
├── PROJECT_EXPLANATION.txt        ← Plain-English technical explainer
├── requirements.txt               ← Python package dependencies
└── README.md                      ← Project documentation (this file)
```

---

## 🚀 Quick Start

### 1. Install Dependencies
Clone the repository and install all dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
Execute the full orchestrator pipeline:
```bash
python src/main.py
```

This single command runs the entire workflow synchronously:
1. Loads parameters from `config/config.yaml`.
2. Synthesizes a messy 3,000-row transaction log.
3. Cleans, imputes, and standardizes data using the 10-step pipeline.
4. Performs ANOVA, Chi-Square tests, and ABC product classifications.
5. Engineers time-series lag features, fits a Random Forest, and forecasts sales.
6. Generates 5 high-resolution dashboards and saves them in `outputs/dashboards/`.
7. Exports text reports in `outputs/reports/` and prints a final CLI summary.

---

## 🧹 10-Step Data Cleaning Pipeline

The `data_cleaner.py` script ensures that data is clean, validated, and formatted before modeling:

1. **Load Data**: Imports the raw CSV dataset and records initial shapes and missing values.
2. **Remove Duplicates**: Drops exact duplicate transactions.
3. **Parse Dates**: Formats transaction timestamps and imputes missing dates with the median date.
4. **Standardize Categoricals**: Maps string variations (e.g. `female`, `FEMALE`, `f` -> `Female`) and strips trailing whitespace.
5. **Handle Invalid Values**: Flags negative transaction quantities (converting them to positive absolute values) and nullifies out-of-range customer ratings.
6. **Impute Missing Values**: Automatically fills numeric gaps with medians and categoricals with modes.
7. **Treat Outliers**: Clips extreme unit prices using the IQR method ($Q3 + 3 \times IQR$).
8. **Recompute Derived Columns**: Recomputes `total_sales` to maintain mathematical consistency (`unit_price * quantity * (1 - discount)`).
9. **Caste Types**: Formats category types and casts IDs/integers correctly.
10. **Validation Suite**: Performs assertions verifying zero nulls, valid rating boundaries, and numeric sanity checks.

### Data Quality Score
A custom **Data Quality Score (0–100)** is computed dynamically based on:
- Row loss percent (penalizing duplicate removal or dropped rows).
- Percent of nulls remaining.
- Frequencies of invalid data points corrected.
- Validated schema compliance.

---

## 📊 Analytical Dashboards

Five high-contrast, dark-mode dashboards are generated:
1. **Overview**: Key business performance metrics (KPI cards), region distributions, category sales, and monthly trends.
2. **Distributions**: Age histograms, price violins, rating scales, and gender distributions.
3. **Correlations**: Correlation heatmap and discount vs. sales scatter charts with regression overlays.
4. **Seasonal Trends**: Quarterly sales volumes, stacked category regional splits, and top 10 products by revenue.
5. **ML Insights & ABC**: Actual vs. predicted sales scatter, 3-month forecast timeline with confidence bands, feature importance rankings, and ABC product category splits.

---

## 🤖 Forecasting Model (Machine Learning)

The forecasting engine in `ml_forecaster.py` treats total daily sales as a regression task:
- **Feature Engineering**:
  - *Lags*: 1-day, 2-day, and 7-day lagged revenue values.
  - *Rolling Windows*: 7-day and 30-day rolling sales averages.
  - *Temporal Features*: Month, quarter, day of the week, and weekend flags.
- **Model**: `RandomForestRegressor` configured through `config/config.yaml`.
- **Validation**: 5-fold Time-Series Cross Validation (`TimeSeriesSplit`) ensuring zero future-to-past data leakage.
- **Output**: Generates predictions for the next 3 months with lower/upper confidence bounds and saves the model to `outputs/models/sales_forecast.pkl`.

---

## 🧪 Running Unit Tests

To run the unit tests for the data cleaning pipeline:
```bash
python -m unittest tests/test_cleaner.py
```
All tests are implemented using Python's standard `unittest` library and require zero configuration.
