"""
report_generator.py
-------------------
Generates two human-readable text reports at the end of the pipeline.

  outputs/reports/data_quality_report.txt  — step-by-step cleaning metrics
  outputs/reports/executive_summary.txt    — business-level insights
"""

import os
from datetime import datetime
from typing import Optional

import pandas as pd

from logger import get_logger

log = get_logger("report_generator")

W = 68   # report width


def _line(ch="─") -> str:
    return ch * W


def generate_data_quality_report(dq_report, report_dir: str) -> str:
    """Persist the DataQualityReport to a text file."""
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "data_quality_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(dq_report.to_text())
    log.info(f"DQ report saved → {os.path.abspath(path)}")
    return path


def generate_executive_summary(
    df:               pd.DataFrame,
    ml_results:       Optional[dict],
    analysis_results: Optional[dict],
    report_dir:       str,
) -> str:
    """Generate a business-level executive summary."""
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, "executive_summary.txt")

    top_region   = (df.groupby("region",           observed=True)["total_sales"]
                    .sum().idxmax())
    top_category = (df.groupby("product_category", observed=True)["total_sales"]
                    .sum().idxmax())
    top_product  = df.groupby("product_name")["total_sales"].sum().idxmax()

    # Year-over-year
    yoy = {yr: df[df["year"] == yr]["total_sales"].sum()
           for yr in sorted(df["year"].unique())}

    with open(path, "w", encoding="utf-8") as f:

        def sec(title):
            f.write(f"\n{_line()}\n  {title}\n{_line()}\n")

        f.write(_line("=") + "\n")
        f.write("  EXECUTIVE SUMMARY  —  Retail Sales Analysis 2023-2024\n")
        f.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(_line("=") + "\n")

        sec("BUSINESS OVERVIEW")
        f.write(f"  Total Revenue       : ₹{df['total_sales'].sum():>16,.2f}\n")
        f.write(f"  Total Orders        : {len(df):>16,}\n")
        f.write(f"  Avg Order Value     : ₹{df['total_sales'].mean():>16,.2f}\n")
        f.write(f"  Avg Customer Rating : {df['customer_rating'].mean():>16.2f} / 5.00\n")
        f.write(f"  Date Range          : "
                f"{df['order_date'].min().date()}  →  "
                f"{df['order_date'].max().date()}\n")
        f.write(f"  Unique Customers    : {df['customer_name'].nunique():>16,}\n")

        sec("TOP PERFORMERS")
        f.write(f"  Top Region          : {top_region}\n")
        f.write(f"  Top Category        : {top_category}\n")
        f.write(f"  Top Product         : {top_product}\n")

        sec("YEAR-OVER-YEAR REVENUE")
        years = sorted(yoy.keys())
        for yr in years:
            f.write(f"  {yr}  :  ₹{yoy[yr]:>14,.2f}\n")
        if len(years) >= 2:
            growth = (yoy[years[-1]] - yoy[years[0]]) / yoy[years[0]] * 100
            f.write(f"  YoY Growth  :  {growth:>+.1f}%\n")

        sec("TOP 5 PRODUCTS BY REVENUE")
        top5 = (df.groupby("product_name")["total_sales"]
                .sum().sort_values(ascending=False).head(5))
        for i, (prod, rev) in enumerate(top5.items(), 1):
            f.write(f"  {i}. {prod:<32}  ₹{rev:>12,.2f}\n")

        sec("REGIONAL PERFORMANCE")
        reg_perf = df.groupby("region", observed=True).agg(
            Revenue=("total_sales",      "sum"),
            Orders =("order_id",         "count"),
            Rating =("customer_rating",  "mean"),
        ).sort_values("Revenue", ascending=False)
        for region, row in reg_perf.iterrows():
            f.write(f"  {region:<12}  Rev=₹{row['Revenue']:>11,.0f}"
                    f"  Orders={row['Orders']:>5,}  AvgRating={row['Rating']:.2f}\n")

        if "loyalty_tier" in df.columns:
            sec("LOYALTY TIER BREAKDOWN")
            lt = df.groupby("loyalty_tier", observed=True).agg(
                Orders  =("order_id",    "count"),
                Revenue =("total_sales", "sum"),
            ).sort_values("Revenue", ascending=False)
            for tier, row in lt.iterrows():
                f.write(f"  {tier:<10}  Orders={row['Orders']:>5,}"
                        f"  Revenue=₹{row['Revenue']:>12,.2f}\n")

        if ml_results:
            m = ml_results.get("metrics", {})
            sec("ML SALES FORECASTING RESULTS")
            f.write(f"  Model            : Random Forest Regressor\n")
            f.write(f"  Test MAE         : ₹{m.get('MAE', 'N/A'):,.2f}\n")
            f.write(f"  Test RMSE        : ₹{m.get('RMSE', 'N/A'):,.2f}\n")
            f.write(f"  Test R²          : {m.get('R2', 'N/A')}\n")
            f.write(f"  CV R² (mean±std) : "
                    f"{m.get('CV_R2_mean','N/A')} ± {m.get('CV_R2_std','N/A')}\n")
            fc = ml_results.get("monthly_forecast", pd.DataFrame())
            if not fc.empty:
                f.write("\n  3-Month Revenue Forecast:\n")
                for _, row in fc.iterrows():
                    f.write(f"    {row['month_label']:<12}  "
                            f"₹{row['predicted_sales']:>12,.2f}  "
                            f"(±₹{row['predicted_sales']*0.15:,.2f})\n")

        if analysis_results:
            abc = analysis_results.get("abc", pd.DataFrame())
            if not abc.empty:
                sec("ABC PRODUCT CLASSIFICATION")
                tc = abc["abc_tier"].value_counts()
                labels = {
                    "A": "generates first 70% of revenue",
                    "B": "next 20% of revenue",
                    "C": "bottom 10% of revenue",
                }
                for tier in ["A", "B", "C"]:
                    f.write(f"  Tier {tier}  ({tc.get(tier,0):>3} products)  "
                            f"—  {labels[tier]}\n")

            repeat = analysis_results.get("repeat", {})
            if repeat:
                sec("CUSTOMER INSIGHTS")
                f.write(f"  Total unique customers   : {repeat.get('total_customers','N/A'):,}\n")
                f.write(f"  Repeat purchase rate     : {repeat.get('repeat_rate_pct','N/A')}%\n")
                f.write(f"  Avg orders per customer  : {repeat.get('avg_orders','N/A')}\n")

        sec("ACTIONABLE RECOMMENDATIONS")
        recs = [
            f"Focus marketing spend on {top_region} — highest revenue region.",
            f"Expand {top_category} inventory — top-grossing category.",
            "Review Electronics return rate — typically highest; look at quality control.",
            "Ramp up inventory in Oct–Dec for festive season Electronics/Clothing surge.",
            "Launch loyalty upgrade campaign: push Silver customers to Gold tier.",
            "Introduce personalised discount offers to repeat customers to increase LTV.",
        ]
        for i, rec in enumerate(recs, 1):
            f.write(f"  {i}. {rec}\n")

        f.write("\n" + _line("=") + "\n")

    log.info(f"Executive summary saved → {os.path.abspath(path)}")
    return path
