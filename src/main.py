"""
main.py  —  v2.0
-----------------
Full pipeline orchestrator:
  1. Load config (config/config.yaml)
  2. Generate raw messy dataset
  3. 10-step data cleaning (v2.0 pipeline)
  4. Statistical analysis (ANOVA, chi-square, ABC, normality)
  5. Generate 4 advanced visual dashboards
  6. ML sales forecasting (Random Forest + 3-month forecast)
  7. Generate text reports (DQ report + executive summary)
  8. Print final console summary
"""

import os
import sys
import io
import time

# ── UTF-8 console output (Windows fix) ───────────────────────────────────────
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Path setup ────────────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, SRC_DIR)

# ── Project imports ───────────────────────────────────────────────────────────
import yaml
from logger           import get_logger
from generate_data    import generate_and_save
from data_cleaner     import run_pipeline
from analyzer         import run_analysis
from visualizer       import generate_all_dashboards
from ml_forecaster    import train_and_forecast
from report_generator import generate_data_quality_report, generate_executive_summary

log = get_logger(
    "main",
    log_file=os.path.join(BASE_DIR, "outputs", "logs", "pipeline.log"),
)

# ── Directory layout ──────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
REPORT_DIR  = os.path.join(OUTPUT_DIR, "reports")
MODEL_DIR   = os.path.join(OUTPUT_DIR, "models")
CHART_DIR   = os.path.join(OUTPUT_DIR, "dashboards")


# ── Config loader ─────────────────────────────────────────────────────────────
def load_config() -> dict:
    cfg_path = os.path.join(BASE_DIR, "config", "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        log.info(f"Config loaded from {cfg_path}")
        return cfg
    log.warning("config.yaml not found — using defaults")
    return {}


# ── Console summary ───────────────────────────────────────────────────────────
def print_final_summary(df, ml_results, dq_report):
    W = 68
    sep  = "=" * W
    sep2 = "-" * W

    print(f"\n{sep}")
    print("  PIPELINE COMPLETE  —  RETAIL SALES ANALYSIS 2023-2024")
    print(sep)

    print(f"\n  Dataset")
    print(f"  {'Shape':<28}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  {'Date range':<28}: {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    print(f"  {'Data Quality Score':<28}: {dq_report.dq_score} / 100")

    print(f"\n  Business KPIs")
    print(f"  {'Total Revenue':<28}: Rs.{df['total_sales'].sum():>14,.2f}")
    print(f"  {'Avg Order Value':<28}: Rs.{df['total_sales'].mean():>14,.2f}")
    print(f"  {'Avg Customer Rating':<28}: {df['customer_rating'].mean():>14.2f} / 5.00")
    top_r = df.groupby("region",           observed=True)["total_sales"].sum().idxmax()
    top_c = df.groupby("product_category", observed=True)["total_sales"].sum().idxmax()
    top_p = df.groupby("product_name")["total_sales"].sum().idxmax()
    print(f"  {'Top Region':<28}: {top_r}")
    print(f"  {'Top Category':<28}: {top_c}")
    print(f"  {'Top Product':<28}: {top_p}")

    if ml_results:
        m = ml_results["metrics"]
        print(f"\n  ML Forecast (Random Forest)")
        print(f"  {'Test R2':<28}: {m['R2']}")
        print(f"  {'Test MAE':<28}: Rs.{m['MAE']:>12,.2f}")
        print(f"  {'Test RMSE':<28}: Rs.{m['RMSE']:>12,.2f}")
        print(f"  {'CV R2 mean +/- std':<28}: {m['CV_R2_mean']} +/- {m['CV_R2_std']}")
        fc = ml_results.get("monthly_forecast")
        if fc is not None and not fc.empty:
            print(f"\n  3-Month Revenue Forecast:")
            for _, row in fc.iterrows():
                print(f"    {row['month_label']:<14}  Rs.{row['predicted_sales']:>12,.0f}"
                      f"  (range Rs.{row['lower']:>10,.0f} - Rs.{row['upper']:>10,.0f})")

    print(f"\n  Outputs")
    print(f"  {'Charts':<28}: {CHART_DIR}")
    print(f"  {'Reports':<28}: {REPORT_DIR}")
    print(f"  {'ML Model':<28}: {MODEL_DIR}")
    print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    log.info("=" * 68)
    log.info("  RETAIL SALES PIPELINE  v2.0  — Starting")
    log.info("=" * 68)

    # ── 1. Config ─────────────────────────────────────────────────────────────
    cfg      = load_config()
    ds_cfg   = cfg.get("dataset",   {})
    cl_cfg   = cfg.get("cleaning",  {})
    ml_cfg   = cfg.get("ml",        {})
    rp_cfg   = cfg.get("reporting", {})

    # ── 2. Generate raw data ──────────────────────────────────────────────────
    log.info("STEP A  Generating raw dataset ...")
    raw_path = generate_and_save()

    # ── 3. Clean data (v2 pipeline — returns 3-tuple) ─────────────────────────
    log.info("STEP B  Running 10-step cleaning pipeline ...")
    df, clean_path, dq_report = run_pipeline(
        raw_path,
        save_dir   = DATA_DIR,
        iqr_factor = cl_cfg.get("outlier_iqr_factor", 3.0),
    )

    # ── 4. Statistical analysis ───────────────────────────────────────────────
    log.info("STEP C  Running statistical analysis ...")
    analysis_results = run_analysis(
        df,
        report_dir = REPORT_DIR,
        abc_a      = rp_cfg.get("abc_a_threshold", 0.70),
        abc_b      = rp_cfg.get("abc_b_threshold", 0.90),
    )

    # ── 5. ML Forecast ────────────────────────────────────────────────────────
    log.info("STEP D  Training ML forecast model ...")
    ml_results = None
    try:
        ml_results = train_and_forecast(
            df,
            model_dir       = MODEL_DIR,
            n_estimators    = ml_cfg.get("n_estimators",    200),
            max_depth       = ml_cfg.get("max_depth",        10),
            cv_folds        = ml_cfg.get("cv_folds",          5),
            forecast_months = ml_cfg.get("forecast_months",   3),
            random_state    = ml_cfg.get("random_state",     42),
        )
    except Exception as e:
        log.error(f"ML step failed (non-fatal): {e}")

    # ── 6. Visualize ──────────────────────────────────────────────────────────
    log.info("STEP E  Generating dashboards ...")
    generate_all_dashboards(
        df,
        CHART_DIR,
        ml_results       = ml_results,
        analysis_results = analysis_results
    )

    # ── 7. Reports ────────────────────────────────────────────────────────────
    log.info("STEP F  Writing text reports ...")
    generate_data_quality_report(dq_report, REPORT_DIR)
    generate_executive_summary(df, ml_results, analysis_results, REPORT_DIR)

    # ── 8. Final summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print_final_summary(df, ml_results, dq_report)
    log.info(f"Total pipeline runtime : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
