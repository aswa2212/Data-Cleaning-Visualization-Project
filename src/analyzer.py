"""
analyzer.py
-----------
Advanced statistical analysis module for the retail sales pipeline.

Analyses performed
  1. Descriptive statistics — mean, median, std, skewness, kurtosis
  2. Normality tests       — Shapiro-Wilk (random 500-row sample)
  3. One-way ANOVA         — does region significantly affect avg sales?
  4. Chi-square test       — is payment method independent of gender?
  5. Pearson correlation   — full matrix with p-values
  6. ABC analysis          — classify products by cumulative revenue
  7. Repeat-customer rate  — via customer_id column
  8. Return rate           — by product category

All results are written to  outputs/reports/statistical_analysis.txt
and returned as a structured dict for use by visualizer.py.
"""

import os
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from logger import get_logger

log = get_logger("analyzer")


# ── 1. Descriptive Statistics ─────────────────────────────────────────────────
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["age", "unit_price", "discount", "total_sales",
            "customer_rating", "quantity", "shipping_days"]
    cols = [c for c in cols if c in df.columns]
    rows = []
    for col in cols:
        s = df[col].dropna()
        rows.append({
            "column":   col,
            "count":    int(len(s)),
            "mean":     round(float(s.mean()),   2),
            "median":   round(float(s.median()), 2),
            "std":      round(float(s.std()),    2),
            "min":      round(float(s.min()),    2),
            "max":      round(float(s.max()),    2),
            "skewness": round(float(stats.skew(s)),     3),
            "kurtosis": round(float(stats.kurtosis(s)), 3),
        })
    return pd.DataFrame(rows).set_index("column")


# ── 2. Normality Tests (Shapiro-Wilk) ────────────────────────────────────────
def normality_tests(df: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    cols = ["age", "unit_price", "total_sales", "customer_rating", "discount"]
    cols = [c for c in cols if c in df.columns]
    rows = []
    for col in cols:
        s      = df[col].dropna()
        sample = s.sample(min(sample_size, len(s)), random_state=42)
        w, p   = stats.shapiro(sample)
        rows.append({
            "column":  col,
            "W_stat":  round(float(w), 4),
            "p_value": round(float(p), 4),
            "normal":  "Yes" if p > 0.05 else "No",
        })
    return pd.DataFrame(rows).set_index("column")


# ── 3. ANOVA — Region vs Sales ────────────────────────────────────────────────
def anova_region_sales(df: pd.DataFrame) -> dict:
    groups = [
        g["total_sales"].values
        for _, g in df.groupby("region", observed=True)
    ]
    f_stat, p_val = stats.f_oneway(*groups)
    region_means  = (df.groupby("region", observed=True)["total_sales"]
                     .mean().round(2))
    return {
        "test":         "One-way ANOVA",
        "variable":     "total_sales grouped by region",
        "F_stat":       round(float(f_stat), 4),
        "p_value":      round(float(p_val),  4),
        "significant":  p_val < 0.05,
        "region_means": region_means.to_dict(),
    }


# ── 4. Chi-Square — Payment Method × Gender ───────────────────────────────────
def chi_square_payment_gender(df: pd.DataFrame) -> dict:
    ct = pd.crosstab(df["payment_method"], df["gender"])
    chi2, p_val, dof, _ = stats.chi2_contingency(ct)
    return {
        "test":        "Chi-square",
        "variables":   "payment_method × gender",
        "chi2_stat":   round(float(chi2), 4),
        "p_value":     round(float(p_val), 4),
        "dof":         int(dof),
        "significant": p_val < 0.05,
    }


# ── 5. Pearson Correlation with P-values ─────────────────────────────────────
def correlation_with_pvalues(df: pd.DataFrame):
    cols = ["age", "unit_price", "discount", "total_sales",
            "customer_rating", "quantity", "shipping_days"]
    cols = [c for c in cols if c in df.columns]
    sub  = df[cols].dropna()
    r_mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    p_mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for c1, c2 in combinations(cols, 2):
        r, p = stats.pearsonr(sub[c1], sub[c2])
        r_mat.loc[c1, c2] = round(r, 3)
        r_mat.loc[c2, c1] = round(r, 3)
        p_mat.loc[c1, c2] = round(p, 4)
        p_mat.loc[c2, c1] = round(p, 4)
    return r_mat, p_mat


# ── 6. ABC Analysis ───────────────────────────────────────────────────────────
def abc_analysis(df: pd.DataFrame,
                 a_thresh: float = 0.70,
                 b_thresh: float = 0.90) -> pd.DataFrame:
    """
    Classify products into:
      Tier A — top products generating first 70% of total revenue
      Tier B — next products up to 90%
      Tier C — remainder
    """
    prod_rev = (df.groupby("product_name")["total_sales"]
                .sum().sort_values(ascending=False))
    cumshare = prod_rev.cumsum() / prod_rev.sum()
    tiers = []
    for cum in cumshare:
        if cum <= a_thresh:
            tiers.append("A")
        elif cum <= b_thresh:
            tiers.append("B")
        else:
            tiers.append("C")
    result = pd.DataFrame({
        "product":   prod_rev.index,
        "revenue":   prod_rev.values.round(2),
        "cum_share": cumshare.values.round(4),
        "abc_tier":  tiers,
    }).reset_index(drop=True)
    return result


# ── 7. Repeat Customer Rate ───────────────────────────────────────────────────
def repeat_customer_analysis(df: pd.DataFrame) -> dict:
    if "customer_id" not in df.columns:
        return {}
    opc = df.groupby("customer_id")["order_id"].count()
    return {
        "total_customers":  int(len(opc)),
        "repeat_customers": int((opc > 1).sum()),
        "repeat_rate_pct":  round((opc > 1).sum() / len(opc) * 100, 2),
        "avg_orders":       round(float(opc.mean()), 2),
        "max_orders":       int(opc.max()),
    }


# ── 8. Return Rate by Category ────────────────────────────────────────────────
def return_rate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    if "return_flag" not in df.columns:
        return pd.DataFrame()
    df2 = df.copy()
    df2["return_flag"] = df2["return_flag"].astype(bool)
    grp = df2.groupby("product_category", observed=True)["return_flag"]
    result = pd.DataFrame({
        "return_count": grp.sum(),
        "total_orders": grp.count(),
    })
    result["return_rate_pct"] = (
        result["return_count"] / result["total_orders"] * 100
    ).round(2)
    return result.sort_values("return_rate_pct", ascending=False)


# ── Master Runner ─────────────────────────────────────────────────────────────
def run_analysis(df: pd.DataFrame,
                 report_dir: str,
                 abc_a: float = 0.70,
                 abc_b: float = 0.90) -> dict:
    """Run all analyses and save results to a text report."""
    os.makedirs(report_dir, exist_ok=True)
    log.info("Running statistical analysis …")

    desc      = descriptive_stats(df)
    norm      = normality_tests(df)
    anova_res = anova_region_sales(df)
    chi_res   = chi_square_payment_gender(df)
    r_mat, p_mat = correlation_with_pvalues(df)
    abc       = abc_analysis(df, a_thresh=abc_a, b_thresh=abc_b)
    repeat    = repeat_customer_analysis(df)
    returns   = return_rate_by_category(df)

    # ── Write report ──────────────────────────────────────────────────────────
    out_path = os.path.join(report_dir, "statistical_analysis.txt")
    W = 68
    with open(out_path, "w", encoding="utf-8") as f:
        def h(title=""):
            f.write(("─" * W + "\n") if not title else
                    f"  {title}\n" + "─" * W + "\n")

        f.write("=" * W + "\n")
        f.write("  STATISTICAL ANALYSIS REPORT  —  Retail Sales Dataset\n")
        f.write("=" * W + "\n\n")

        h("1. DESCRIPTIVE STATISTICS")
        f.write(desc.to_string()); f.write("\n\n")

        h("2. NORMALITY TESTS  (Shapiro-Wilk, sample n=500)")
        f.write(norm.to_string()); f.write("\n\n")

        h("3. ONE-WAY ANOVA  —  Total Sales across Regions")
        f.write(f"  F-statistic  : {anova_res['F_stat']}\n")
        f.write(f"  p-value      : {anova_res['p_value']}\n")
        sig = ("YES — region significantly affects avg sales"
               if anova_res["significant"] else "NO — no significant difference")
        f.write(f"  Significant  : {sig}\n")
        f.write("  Regional mean sales:\n")
        for reg, mean in sorted(anova_res["region_means"].items(),
                                key=lambda x: x[1], reverse=True):
            f.write(f"    {reg:<12}  ₹{mean:>12,.2f}\n")
        f.write("\n")

        h("4. CHI-SQUARE TEST  —  Payment Method × Gender")
        f.write(f"  χ² statistic      : {chi_res['chi2_stat']}\n")
        f.write(f"  p-value           : {chi_res['p_value']}\n")
        f.write(f"  Degrees of freedom: {chi_res['dof']}\n")
        sig2 = ("YES — payment choice and gender are associated"
                if chi_res["significant"] else "NO — independent at α=0.05")
        f.write(f"  Significant       : {sig2}\n\n")

        h("5. PEARSON CORRELATION MATRIX  (r values)")
        f.write(r_mat.to_string()); f.write("\n\n")

        h("6. ABC PRODUCT ANALYSIS")
        tier_counts = abc["abc_tier"].value_counts()
        labels = {"A": "top 70% revenue", "B": "next 20%", "C": "bottom 10%"}
        for tier in ["A", "B", "C"]:
            f.write(f"  Tier {tier}  ({labels[tier]})  :  "
                    f"{tier_counts.get(tier, 0)} products\n")
        f.write("\n")
        f.write(abc.to_string(index=False)); f.write("\n\n")

        if repeat:
            h("7. REPEAT CUSTOMER ANALYSIS")
            for k, v in repeat.items():
                f.write(f"  {k:<28} : {v}\n")
            f.write("\n")

        if not returns.empty:
            h("8. RETURN RATE BY CATEGORY")
            f.write(returns.to_string()); f.write("\n")

        f.write("\n" + "=" * W + "\n")

    log.info(f"Statistical analysis saved → {os.path.abspath(out_path)}")

    return {
        "descriptive":  desc,
        "normality":    norm,
        "anova":        anova_res,
        "chi_square":   chi_res,
        "correlations": (r_mat, p_mat),
        "abc":          abc,
        "repeat":       repeat,
        "returns":      returns,
    }
