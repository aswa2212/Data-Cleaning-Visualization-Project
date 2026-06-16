"""
ml_forecaster.py
----------------
Machine learning module: trains a Random Forest on daily sales data,
evaluates it with time-series cross-validation, and forecasts the next
3 months of revenue.

Pipeline
  1. Aggregate orders → daily total sales time series
  2. Feature engineering (lags, rolling stats, time encodings)
  3. Time-based train/test split (80/20)
  4. Random Forest training
  5. 5-fold TimeSeriesSplit cross-validation (R² metric)
  6. Test evaluation: MAE, RMSE, R²
  7. 3-month daily forecast → aggregated to monthly
  8. Feature importance extraction
  9. Persist trained model to  outputs/models/sales_forecast.pkl
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder

from logger import get_logger

log = get_logger("ml_forecaster")

# Features used for training
_FEATURE_COLS = [
    "month", "quarter", "year",
    "day_of_week", "day_of_month",
    "is_weekend", "is_festive_month",
    "lag_1", "lag_2", "lag_3", "lag_7",
    "rolling_7_mean", "rolling_7_std",
    "rolling_30_mean",
    "price_mean", "discount_mean", "quantity_mean",
    "region_enc", "category_enc",
]


# ── Step 1 — Daily Aggregation ────────────────────────────────────────────────
def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("order_date")
        .agg(
            total_sales   = ("total_sales",  "sum"),
            n_orders      = ("order_id",     "count"),
            price_mean    = ("unit_price",   "mean"),
            discount_mean = ("discount",     "mean"),
            quantity_mean = ("quantity",     "mean"),
        )
        .reset_index()
        .sort_values("order_date")
    )
    # Fill calendar gaps with zero-sales days
    full_range = pd.date_range(
        daily["order_date"].min(),
        daily["order_date"].max(),
        freq="D",
    )
    daily = (daily.set_index("order_date")
             .reindex(full_range, fill_value=0)
             .reset_index()
             .rename(columns={"index": "order_date"}))
    return daily


# ── Step 2 — Feature Engineering ─────────────────────────────────────────────
def _engineer_features(
    daily: pd.DataFrame,
    df_orig: pd.DataFrame,
) -> tuple:
    d = daily.copy()
    d["order_date"]       = pd.to_datetime(d["order_date"])
    d["month"]            = d["order_date"].dt.month
    d["quarter"]          = d["order_date"].dt.quarter
    d["year"]             = d["order_date"].dt.year
    d["day_of_week"]      = d["order_date"].dt.dayofweek
    d["day_of_month"]     = d["order_date"].dt.day
    d["is_weekend"]       = (d["day_of_week"] >= 5).astype(int)
    d["is_festive_month"] = d["month"].isin([10, 11, 12]).astype(int)

    # Lag features (shift prevents data leakage)
    for lag in [1, 2, 3, 7]:
        d[f"lag_{lag}"] = d["total_sales"].shift(lag)

    # Rolling features (based on shifted series — no leakage)
    shifted = d["total_sales"].shift(1)
    d["rolling_7_mean"]  = shifted.rolling(7,  min_periods=1).mean()
    d["rolling_7_std"]   = shifted.rolling(7,  min_periods=1).std().fillna(0)
    d["rolling_30_mean"] = shifted.rolling(30, min_periods=1).mean()

    # Dominant category per day (label-encoded)
    df_orig = df_orig.copy()
    df_orig["order_date"] = pd.to_datetime(df_orig["order_date"])
    top_cat = (df_orig.groupby(["order_date", "product_category"], observed=True)
               ["total_sales"].sum().reset_index()
               .sort_values("total_sales", ascending=False)
               .drop_duplicates("order_date"))
    le_cat = LabelEncoder()
    top_cat["category_enc"] = le_cat.fit_transform(
        top_cat["product_category"].astype(str))
    d = d.merge(top_cat[["order_date", "category_enc"]], on="order_date", how="left")

    # Dominant region per day (label-encoded)
    top_reg = (df_orig.groupby(["order_date", "region"], observed=True)
               ["total_sales"].sum().reset_index()
               .sort_values("total_sales", ascending=False)
               .drop_duplicates("order_date"))
    le_reg = LabelEncoder()
    top_reg["region_enc"] = le_reg.fit_transform(
        top_reg["region"].astype(str))
    d = d.merge(top_reg[["order_date", "region_enc"]], on="order_date", how="left")

    d = d.fillna(0)
    return d, le_cat, le_reg


# ── Main Training & Forecasting Function ─────────────────────────────────────
def train_and_forecast(
    df:               pd.DataFrame,
    model_dir:        str,
    n_estimators:     int   = 200,
    max_depth:        int   = 10,
    cv_folds:         int   = 5,
    forecast_months:  int   = 3,
    random_state:     int   = 42,
) -> dict:
    """
    Train a Random Forest, cross-validate, forecast, and save the model.

    Returns a rich results dict consumed by visualizer.py Dashboard 5.
    """
    log.info("Starting ML forecasting pipeline …")

    # 1. Aggregate to daily
    df["order_date"] = pd.to_datetime(df["order_date"])
    daily = _aggregate_daily(df)
    log.info(f"Daily time series: {len(daily)} days")

    # 2. Feature engineering
    featured, le_cat, le_reg = _engineer_features(daily, df)
    feat_cols = [c for c in _FEATURE_COLS if c in featured.columns]
    featured  = featured.dropna(subset=feat_cols + ["total_sales"])

    X     = featured[feat_cols].values
    y     = featured["total_sales"].values
    dates = featured["order_date"].values

    # 3. Train/test split (time-ordered, 80/20 — no shuffle)
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 4. Train Random Forest
    model = RandomForestRegressor(
        n_estimators   = n_estimators,
        max_depth      = max_depth,
        min_samples_leaf = 5,
        n_jobs         = -1,
        random_state   = random_state,
    )
    model.fit(X_train, y_train)
    log.info(f"Trained on {len(X_train):,} days  |  test on {len(X_test):,} days")

    # 5. TimeSeriesSplit cross-validation
    tscv       = TimeSeriesSplit(n_splits=cv_folds)
    cv_r2      = cross_val_score(model, X, y, cv=tscv, scoring="r2")
    log.info(f"CV R² per fold : {[round(s,3) for s in cv_r2]}")
    log.info(f"Mean CV R²     : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # 6. Test metrics
    y_pred = model.predict(X_test)
    mae    = float(mean_absolute_error(y_test, y_pred))
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2     = float(r2_score(y_test, y_pred))
    log.info(f"Test  MAE={mae:,.2f}  RMSE={rmse:,.2f}  R²={r2:.4f}")

    # 7. Feature importance
    importance = dict(sorted(
        zip(feat_cols, model.feature_importances_.tolist()),
        key=lambda x: x[1], reverse=True,
    ))

    # 8. Future forecast (next forecast_months × 30 days)
    last_date    = pd.to_datetime(dates[-1])
    future_dates = pd.date_range(
        start  = last_date + pd.Timedelta(days=1),
        periods = forecast_months * 30,
        freq   = "D",
    )
    rolling_buf = list(daily["total_sales"].values[-30:])

    forecast_rows = []
    for fd in future_dates:
        row = {
            "month":            fd.month,
            "quarter":          fd.quarter,
            "year":             fd.year,
            "day_of_week":      fd.dayofweek,
            "day_of_month":     fd.day,
            "is_weekend":       int(fd.dayofweek >= 5),
            "is_festive_month": int(fd.month in [10, 11, 12]),
            "lag_1":            rolling_buf[-1],
            "lag_2":            rolling_buf[-2],
            "lag_3":            rolling_buf[-3],
            "lag_7":            rolling_buf[-7] if len(rolling_buf) >= 7 else rolling_buf[0],
            "rolling_7_mean":   float(np.mean(rolling_buf[-7:])),
            "rolling_7_std":    float(np.std(rolling_buf[-7:])),
            "rolling_30_mean":  float(np.mean(rolling_buf[-30:])),
            "price_mean":       float(df["unit_price"].median()),
            "discount_mean":    float(df["discount"].median()),
            "quantity_mean":    float(df["quantity"].median()),
            "region_enc":       0,
            "category_enc":     0,
        }
        x_row   = np.array([[row[c] for c in feat_cols]])
        pred    = max(0.0, float(model.predict(x_row)[0]))
        forecast_rows.append({"date": fd, "predicted_sales": pred})
        rolling_buf.append(pred)
        if len(rolling_buf) > 30:
            rolling_buf.pop(0)

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df["month_label"] = forecast_df["date"].dt.strftime("%b %Y")
    monthly_fc = (forecast_df.groupby("month_label", sort=False)["predicted_sales"]
                  .sum().reset_index())
    monthly_fc["lower"] = monthly_fc["predicted_sales"] * 0.85
    monthly_fc["upper"] = monthly_fc["predicted_sales"] * 1.15

    # 9. Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sales_forecast.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump({
            "model":    model,
            "features": feat_cols,
            "le_cat":   le_cat,
            "le_reg":   le_reg,
        }, fh)
    log.info(f"Model saved → {os.path.abspath(model_path)}")

    evaluation_df = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred
    })
    
    daily_historical = pd.DataFrame({
        "order_date": pd.to_datetime(dates),
        "total_sales": y
    })
    
    daily_forecast = pd.DataFrame(forecast_rows)
    daily_forecast = daily_forecast.rename(columns={"date": "order_date"})
    daily_forecast["lower"] = daily_forecast["predicted_sales"] * 0.85
    daily_forecast["upper"] = daily_forecast["predicted_sales"] * 1.15

    return {
        "model":               model,
        "feature_cols":        feat_cols,
        "feature_importances": importance,
        "train_dates":         dates[:split],
        "test_dates":          dates[split:],
        "actual":              y_test,
        "predicted":           y_pred,
        "evaluation_df":       evaluation_df,
        "daily_historical":    daily_historical,
        "daily_forecast":      daily_forecast,
        "all_dates":           dates,
        "all_actual":          y,
        "metrics": {
            "MAE":          round(mae,              2),
            "RMSE":         round(rmse,             2),
            "R2":           round(r2,               4),
            "CV_R2_mean":   round(float(cv_r2.mean()), 4),
            "CV_R2_std":    round(float(cv_r2.std()),  4),
        },
        "monthly_forecast": monthly_fc,
        "model_path":       model_path,
    }
