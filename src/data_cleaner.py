"""
data_cleaner.py  —  v2.0
-------------------------
Production-grade 10-step data cleaning pipeline with structured
DataQualityReport output, a DQ score (0-100), and proper logging.

Bug-fixes vs v1
  ✓ total_sales removed from outlier_cols  (it is recomputed in Step 8)
  ✓ Step ordering fixed: imputation (Step 6) now runs BEFORE outlier
    capping (Step 7) — prevents re-introduction of nulls in key cols
  ✓ GENDER_MAP: unknown values are logged and kept as-is, not silently
    converted to NaN
  ✓ scipy.stats now used meaningfully (Z-score informational check)

New features
  ✓ DataQualityReport dataclass — accumulates per-step metrics
  ✓ step_validate() — final schema + null assertions with PASS/FAIL
  ✓ DQ Score (0-100) computed from row loss, null rate, and issue counts
  ✓ run_pipeline() returns (df, path, DataQualityReport) tuple
  ✓ All logging via logger.py (no raw print() calls)
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from logger import get_logger

log = get_logger("data_cleaner")


# ══════════════════════════════════════════════════════════════════════════════
#  DataQualityReport
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DataQualityReport:
    # Raw snapshot
    raw_rows:         int = 0
    raw_cols:         int = 0
    raw_nulls:        int = 0
    raw_duplicates:   int = 0

    # Issues found & fixed
    duplicates_removed:    int = 0
    invalid_dates:         int = 0
    gender_variants_fixed: int = 0
    unknown_genders:       int = 0
    negative_qty_fixed:    int = 0
    bad_ratings_nulled:    int = 0
    whitespace_cols:       int = 0
    outliers_capped:       dict = field(default_factory=dict)
    imputed:               dict = field(default_factory=dict)

    # Final state
    final_rows:   int   = 0
    final_cols:   int   = 0
    final_nulls:  int   = 0
    dq_score:     float = 0.0
    failed_checks: list = field(default_factory=list)

    def compute_score(self) -> float:
        """Heuristic DQ score 0–100. Penalises remaining nulls,
        row loss > 5%, and each category of issues found."""
        score = 100.0

        # Penalise row loss
        if self.raw_rows > 0:
            row_loss_pct = (self.raw_rows - self.final_rows) / self.raw_rows
            score -= min(20.0, row_loss_pct * 200)

        # Penalise remaining nulls
        total_cells = self.final_rows * max(self.final_cols, 1)
        if total_cells > 0:
            null_pct = self.final_nulls / total_cells
            score -= min(30.0, null_pct * 1_000)

        # Penalise each issue type
        deductions = [
            self.invalid_dates         * 0.5,
            self.negative_qty_fixed    * 0.3,
            self.bad_ratings_nulled    * 0.3,
            self.gender_variants_fixed * 0.05,
            len(self.failed_checks)    * 5.0,
        ]
        score -= min(25.0, sum(deductions))

        self.dq_score = round(max(0.0, min(100.0, score)), 1)
        return self.dq_score

    def to_text(self) -> str:
        lines = [
            "=" * 64,
            "  DATA QUALITY REPORT  —  Retail Sales Pipeline v2.0",
            "=" * 64,
            f"  Raw shape          : {self.raw_rows:,} rows × {self.raw_cols} columns",
            f"  Raw nulls          : {self.raw_nulls:,}",
            f"  Raw duplicates     : {self.raw_duplicates:,}",
            "",
            "  ── Issues Detected & Fixed ────────────────────────────",
            f"  Duplicates removed      : {self.duplicates_removed:,}",
            f"  Invalid dates replaced  : {self.invalid_dates:,}",
            f"  Gender variants fixed   : {self.gender_variants_fixed:,}",
            f"  Unknown genders kept    : {self.unknown_genders:,}",
            f"  Negative qty corrected  : {self.negative_qty_fixed:,}",
            f"  Out-of-range ratings    : {self.bad_ratings_nulled:,}",
            f"  Whitespace cols cleaned : {self.whitespace_cols:,}",
        ]
        for col, n in self.outliers_capped.items():
            lines.append(f"  Outliers capped [{col}]  : {n:,}")
        for col, (n, val) in self.imputed.items():
            lines.append(f"  Imputed [{col}]  : {n:,} with {val}")
        lines += [
            "",
            "  ── Final State ────────────────────────────────────────",
            f"  Final shape        : {self.final_rows:,} rows × {self.final_cols} columns",
            f"  Remaining nulls    : {self.final_nulls:,}",
        ]
        if self.failed_checks:
            lines.append(f"  Failed checks      : {', '.join(self.failed_checks)}")
        lines += [
            f"  Data Quality Score : {self.dq_score} / 100",
            "=" * 64,
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════
GENDER_MAP = {
    "male": "Male", "m": "Male", "MALE": "Male", "M": "Male",
    "female": "Female", "f": "Female", "FEMALE": "Female", "F": "Female",
}

W = 64   # separator width


def _sep(title: str = "") -> None:
    if title:
        pad = max(1, (W - len(title) - 2) // 2)
        log.info("─" * pad + f" {title} " + "─" * pad)
    else:
        log.info("─" * W)


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline Steps
# ══════════════════════════════════════════════════════════════════════════════

def step_load(path: str, report: DataQualityReport) -> pd.DataFrame:
    _sep("STEP 1 · LOAD")
    df = pd.read_csv(path)
    report.raw_rows       = len(df)
    report.raw_cols       = df.shape[1]
    report.raw_nulls      = int(df.isnull().sum().sum())
    report.raw_duplicates = int(df.duplicated().sum())
    log.info(f"Loaded      : {df.shape[0]:,} rows × {df.shape[1]} columns")
    log.info(f"Nulls       : {report.raw_nulls:,} total missing values")
    log.info(f"Duplicates  : {report.raw_duplicates:,} exact duplicate rows")
    return df


def step_remove_duplicates(df: pd.DataFrame,
                            report: DataQualityReport) -> pd.DataFrame:
    _sep("STEP 2 · DUPLICATES")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    report.duplicates_removed = removed
    log.info(f"Removed {removed:,} duplicate rows  ({before:,} → {len(df):,})")
    return df


def step_fix_dates(df: pd.DataFrame,
                   report: DataQualityReport) -> pd.DataFrame:
    _sep("STEP 3 · DATE PARSING")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    invalid = int(df["order_date"].isna().sum())
    report.invalid_dates = invalid
    valid_dates = df["order_date"].dropna().sort_values()
    median_date = valid_dates.iloc[len(valid_dates) // 2]
    df["order_date"] = df["order_date"].fillna(median_date)
    log.info(f"Invalid dates found  : {invalid}")
    log.info(f"Filled               : {invalid} rows with median date ({median_date.date()})")
    return df


def step_standardise_categoricals(df: pd.DataFrame,
                                   report: DataQualityReport) -> pd.DataFrame:
    _sep("STEP 4 · STANDARDISE CATEGORICALS")

    # ── Gender ────────────────────────────────────────────────────────────────
    before_vals      = sorted(df["gender"].dropna().unique().tolist())
    known_fixed      = 0
    unknown_kept     = 0

    def _map_gender(x):
        nonlocal known_fixed, unknown_kept
        if pd.isna(x):
            return x
        s = str(x).strip()
        mapped = GENDER_MAP.get(s)
        if mapped:
            known_fixed += 1
            return mapped
        if s in ("Male", "Female"):
            return s
        # Unknown value — keep as-is, just count it
        unknown_kept += 1
        return s

    df["gender"] = df["gender"].map(_map_gender)
    report.gender_variants_fixed = known_fixed
    report.unknown_genders       = unknown_kept
    after_vals = sorted(df["gender"].dropna().unique().tolist())
    log.info(f"Gender before : {before_vals}")
    log.info(f"Gender after  : {after_vals}")
    if unknown_kept:
        log.warning(f"{unknown_kept} unknown gender values kept as-is")

    # ── Whitespace in all text cols ───────────────────────────────────────────
    str_cols = df.select_dtypes(include="object").columns.tolist()
    for col in str_cols:
        df[col] = df[col].str.strip()
    report.whitespace_cols = len(str_cols)
    log.info(f"Stripped whitespace from {len(str_cols)} text columns")

    # ── Normalise loyalty_tier capitalisation ─────────────────────────────────
    if "loyalty_tier" in df.columns:
        df["loyalty_tier"] = df["loyalty_tier"].str.title()

    return df


def step_fix_invalid_values(df: pd.DataFrame,
                             report: DataQualityReport) -> pd.DataFrame:
    _sep("STEP 5 · INVALID VALUES")

    # Negative quantities → absolute value (return entries)
    neg_qty = int((df["quantity"] < 0).sum())
    df["quantity"] = df["quantity"].abs()
    report.negative_qty_fixed = neg_qty
    log.info(f"Negative quantities fixed : {neg_qty}")

    # Out-of-range ratings [1.0, 5.0] → null (re-imputed in Step 6)
    bad_mask = (
        ~df["customer_rating"].between(1.0, 5.0, inclusive="both")
        & df["customer_rating"].notna()
    )
    bad_ratings = int(bad_mask.sum())
    df.loc[bad_mask, "customer_rating"] = np.nan
    report.bad_ratings_nulled = bad_ratings
    log.info(f"Out-of-range ratings nulled : {bad_ratings}")

    return df


def step_impute_missing(df: pd.DataFrame,
                         report: DataQualityReport) -> pd.DataFrame:
    """Step 6 — imputation BEFORE outlier capping (fixes step-ordering bug)."""
    _sep("STEP 6 · IMPUTE MISSING VALUES")

    num_cols = ["age", "unit_price", "discount", "customer_rating", "shipping_days"]
    for col in num_cols:
        if col not in df.columns:
            continue
        n_miss = int(df[col].isna().sum())
        if n_miss:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            report.imputed[col] = (n_miss, f"median={median:.2f}")
            log.info(f"{col:<22} → filled {n_miss:,} with median ({median:.2f})")

    cat_cols = ["gender", "payment_method", "region", "city", "state", "loyalty_tier"]
    for col in cat_cols:
        if col not in df.columns:
            continue
        n_miss = int(df[col].isna().sum())
        if n_miss:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            report.imputed[col] = (n_miss, f"mode='{mode_val}'")
            log.info(f"{col:<22} → filled {n_miss:,} with mode ('{mode_val}')")

    remaining = int(df.isnull().sum().sum())
    log.info(f"Remaining nulls after imputation : {remaining}")
    return df


def step_remove_outliers(df: pd.DataFrame,
                          report: DataQualityReport,
                          iqr_factor: float = 3.0) -> pd.DataFrame:
    """Step 7 — IQR capping on unit_price ONLY.
    BUG FIX: total_sales is excluded because it is fully recomputed in Step 8."""
    _sep("STEP 7 · OUTLIER TREATMENT (IQR × 3)")
    outlier_cols = ["unit_price"]   # total_sales intentionally excluded
    for col in outlier_cols:
        if col not in df.columns:
            continue
        q1, q3  = df[col].quantile([0.25, 0.75])
        iqr     = q3 - q1
        lo, hi  = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        n_out   = int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lower=lo, upper=hi)
        report.outliers_capped[col] = n_out
        log.info(f"{col:<15} outliers capped : {n_out}  (range [{lo:,.0f}, {hi:,.0f}])")

    # Z-score informational check on age (not capped — just logged)
    z_age = np.abs(stats.zscore(df["age"].dropna()))
    log.debug(f"age: {int((z_age > 3).sum())} values |Z|>3 (informational, not capped)")

    return df


def step_recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    _sep("STEP 8 · RECOMPUTE DERIVED COLUMNS")
    df["total_sales"] = (
        df["unit_price"] * df["quantity"] * (1 - df["discount"])
    ).round(2)
    df["year"]       = df["order_date"].dt.year
    df["month"]      = df["order_date"].dt.month
    df["month_name"] = df["order_date"].dt.strftime("%b")
    df["quarter"]    = df["order_date"].dt.quarter
    log.info("Recomputed: total_sales, year, month, month_name, quarter")
    return df


def step_type_cast(df: pd.DataFrame) -> pd.DataFrame:
    _sep("STEP 9 · TYPE CASTING")
    df["age"]      = df["age"].astype(int)
    df["quantity"] = df["quantity"].astype(int)
    if "shipping_days" in df.columns:
        df["shipping_days"] = df["shipping_days"].round().astype(int)
    cat_cols = ["product_category", "region", "gender",
                "payment_method", "loyalty_tier"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    log.info("age, quantity, shipping_days → int  |  categoricals → category dtype")
    return df


def step_validate(df: pd.DataFrame,
                  report: DataQualityReport) -> bool:
    """Step 10 — Assert data invariants. Returns True if all checks pass."""
    _sep("STEP 10 · VALIDATION")
    checks = {
        "No nulls in key columns":
            df[["age", "quantity", "unit_price", "total_sales", "order_date"]]
            .isnull().sum().sum() == 0,
        "All ratings in [1.0, 5.0]":
            df["customer_rating"].between(1.0, 5.0).all(),
        "All quantities >= 0":
            (df["quantity"] >= 0).all(),
        "All unit prices > 0":
            (df["unit_price"] > 0).all(),
        "Discount in [0, 1]":
            df["discount"].between(0.0, 1.0).all(),
        "No empty gender values":
            df["gender"].notna().all(),
    }

    all_pass = True
    for check, result in checks.items():
        status = "PASS ✓" if result else "FAIL ✗"
        log.info(f"  [{status}]  {check}")
        if not result:
            all_pass = False
            report.failed_checks.append(check)

    report.final_rows  = len(df)
    report.final_cols  = df.shape[1]
    report.final_nulls = int(df.isnull().sum().sum())
    report.compute_score()
    log.info(f"Data Quality Score : {report.dq_score} / 100")
    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    raw_path: str,
    save_dir: Optional[str] = None,
    iqr_factor: float = 3.0,
) -> tuple:
    """
    Execute the full 10-step cleaning pipeline.

    Returns
    -------
    (cleaned_df, out_csv_path, DataQualityReport)
    """
    log.info("═" * 64)
    log.info("  DATA CLEANING PIPELINE  —  v2.0")
    log.info("═" * 64)

    report = DataQualityReport()

    df = step_load(raw_path, report)
    df = step_remove_duplicates(df, report)
    df = step_fix_dates(df, report)
    df = step_standardise_categoricals(df, report)
    df = step_fix_invalid_values(df, report)
    df = step_impute_missing(df, report)          # ← before outlier cap
    df = step_remove_outliers(df, report, iqr_factor)
    df = step_recompute_derived(df)
    df = step_type_cast(df)
    step_validate(df, report)

    _sep("CLEANING COMPLETE")
    log.info(f"Final shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    log.info(f"Nulls left   : {df.isnull().sum().sum()}")
    log.info(f"DQ Score     : {report.dq_score} / 100")

    if save_dir is None:
        save_dir = os.path.dirname(raw_path)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "cleaned_sales_data.csv")
    df.to_csv(out_path, index=False)
    log.info(f"Saved cleaned CSV → {os.path.abspath(out_path)}")

    return df, out_path, report


if __name__ == "__main__":
    raw = os.path.join(os.path.dirname(__file__), "..", "data", "raw_sales_data.csv")
    df, path, rpt = run_pipeline(raw)
    print(rpt.to_text())
