"""
visualizer.py  —  v2.0 (Redesigned)
-----------------------------------
Produces 5 publication-quality dark-mode dashboard PNGs.

Dashboard 1 – Overview       : GridSpec KPI cards, region bar, category donut, monthly trend
Dashboard 2 – Distributions  : Age hist+KDE, price violin, rating bar, discount, gender donut, loyalty
Dashboard 3 – Correlations   : Heatmap, discount–sales scatter (R²), payment bars, rating boxplot
Dashboard 4 – Trends         : Quarterly bars (YoY%), stacked region-category, top-10 products, payment pie
Dashboard 5 – ML Insights    : Feature importance, actual vs predicted, 3-month forecast, return rates
"""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── Design System (Premium Corporate Light Palette) ───────────────────────────
P = {
    "bg":      "#F8FAFC",   # Clean light slate background
    "surface": "#FFFFFF",   # Subplot card area background
    "card":    "#FFFFFF",   # Clean white cards
    "border":  "#E2E8F0",   # Soft border grey
    "a1":      "#2563EB",   # Royal Blue (Primary Accent)
    "a2":      "#4F46E5",   # Corporate Indigo
    "a3":      "#0D9488",   # Business Teal
    "a4":      "#10B981",   # Emerald green
    "a5":      "#D97706",   # Business Amber
    "text":    "#0F172A",   # Deep slate-900 text
    "sub":     "#475569",   # slate-600 professional muted text
}

CAT_COLORS = [P["a1"], P["a2"], P["a3"], P["a4"], P["a5"], "#0891B2"]
REGION_COLORS = [P["a1"], P["a2"], P["a3"], P["a4"], P["a5"]]
MONTH_ORDER   = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

CURRENCY = FuncFormatter(lambda x, _: f"₹{x/1_000:.0f}K" if x >= 1_000 else f"₹{x:.0f}")
MILLION  = FuncFormatter(lambda x, _: f"₹{x/1e6:.2f}M")


def _style():
    plt.rcParams.update({
        "figure.facecolor":  P["bg"],
        "axes.facecolor":    P["surface"],
        "axes.edgecolor":    P["border"],
        "axes.labelcolor":   P["text"],
        "axes.titlecolor":   P["text"],
        "xtick.color":       P["sub"],
        "ytick.color":       P["sub"],
        "text.color":        P["text"],
        "grid.color":        P["border"],
        "grid.linestyle":    ":",
        "grid.alpha":        0.30,
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Segoe UI", "Inter", "Helvetica", "Arial", "sans-serif"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _card(ax, title=""):
    """Styles an axes as a clean glass card container without box borders."""
    ax.set_facecolor(P["card"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(P["border"])
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(colors=P["sub"], labelsize=9)
    if title:
        ax.set_title(title, loc="left", pad=15, fontsize=12, fontweight="bold", color=P["text"])


def _title(fig, title, sub=""):
    """Left-aligns the dashboard titles for a clean, editorial layout."""
    fig.text(0.05, 0.965, title, ha="left", va="top",
             fontsize=24, fontweight="bold", color=P["text"])
    if sub:
        fig.text(0.05, 0.938, sub, ha="left", va="top",
                 fontsize=11, color=P["sub"])


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def dashboard_overview(df: pd.DataFrame, out_path: str):
    _style()
    fig = plt.figure(figsize=(22, 15), facecolor=P["bg"])
    fig.subplots_adjust(top=0.86, bottom=0.06, left=0.05, right=0.96,
                        hspace=0.42, wspace=0.32)
    _title(fig, "📊  Sales Overview Dashboard",
           "Retail performance metrics, regional highlights, category splits, and time-series trends (2023 – 2024)")

    # 3 rows: Row 0 (KPI Cards), Row 1 & 2 (Visualizations)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[0.24, 1.0, 1.0])

    # ── KPI Cards (Grid-based layout with left accent bars) ───────────────────
    kpis = [
        ("Total Revenue",    f"₹{df['total_sales'].sum()/1e6:.2f}M",  P["a4"]),
        ("Total Orders",     f"{len(df):,}",                          P["a1"]),
        ("Avg Order Value",  f"₹{df['total_sales'].mean():,.0f}",     P["a5"]),
        ("Avg Rating",       f"{df['customer_rating'].mean():.2f} ★", P["a2"]),
        ("Unique Customers", f"{df['customer_name'].nunique():,}",    P["a3"]),
    ]
    
    gs_kpi = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :], wspace=0.08)
    for i, (label, val, col) in enumerate(kpis):
        ax_k = fig.add_subplot(gs_kpi[0, i])
        ax_k.set_facecolor(P["card"])
        for spine in ax_k.spines.values():
            spine.set_visible(False)
        ax_k.set_xticks([]); ax_k.set_yticks([])
        
        # Neon vertical accent line on left edge
        ax_k.axvline(x=0.02, color=col, linewidth=4, ymin=0.15, ymax=0.85)
        
        ax_k.text(0.12, 0.60, val, fontsize=20, fontweight="bold", color=col, transform=ax_k.transAxes, va="center")
        ax_k.text(0.12, 0.30, label, fontsize=9.5, color=P["sub"], transform=ax_k.transAxes, va="center")

    # ── Sales by Region ───────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    _card(ax1, "Sales by Region")
    reg = df.groupby("region", observed=True)["total_sales"].sum().sort_values()
    bars = ax1.barh(reg.index, reg.values,
                    color=REGION_COLORS[:len(reg)], edgecolor="none", height=0.55)
    for bar, val in zip(bars, reg.values):
        ax1.text(val + reg.max()*0.02, bar.get_y() + bar.get_height()/2,
                 f"₹{val/1e6:.2f}M", va="center", fontsize=9.5, color=P["text"])
    ax1.xaxis.set_major_formatter(CURRENCY)
    ax1.set_xlabel("Total Sales")
    ax1.grid(axis="x")

    # ── Category Donut (Upgraded from Pie) ────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    _card(ax2, "Revenue by Category")
    cat = df.groupby("product_category", observed=True)["total_sales"].sum()
    
    # Hollow donut wedge
    wedges, texts, autotexts = ax2.pie(
        cat.values, labels=cat.index,
        colors=CAT_COLORS[:len(cat)],
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": P["bg"], "linewidth": 3, "width": 0.35},
        textprops={"color": P["text"], "fontsize": 9.5},
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")
    
    # Total center value
    ax2.text(0, 0, f"₹{cat.sum()/1e6:.1f}M\nTotal", ha="center", va="center",
             fontsize=12, fontweight="bold", color=P["text"])

    # ── Monthly Sales Trend (Clean smooth line + Area fill) ───────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    _card(ax3, "Monthly Sales Trend")
    monthly = (df.groupby(["year", "month_name"], observed=True)["total_sales"]
               .sum().reset_index())
    monthly["month_name"] = pd.Categorical(monthly["month_name"],
                                           categories=MONTH_ORDER, ordered=True)
    colors_yr = [P["a1"], P["a3"]]
    for i, (yr, grp) in enumerate(monthly.groupby("year")):
        grp = grp.sort_values("month_name")
        ax3.plot(grp["month_name"].astype(str), grp["total_sales"],
                 marker="o", markersize=6, lw=3,
                 color=colors_yr[i % 2], label=str(yr))
        ax3.fill_between(grp["month_name"].astype(str), grp["total_sales"],
                         alpha=0.08, color=colors_yr[i % 2])
    ax3.yaxis.set_major_formatter(CURRENCY)
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Sales")
    ax3.legend(title="Year", fontsize=9, loc="upper right")
    ax3.grid(axis="y")
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=8.5)

    # ── Loyalty Tier Bar ──────────────────────────────────────────────────────
    if "loyalty_tier" in df.columns:
        ax4 = fig.add_subplot(gs[2, 0])
        _card(ax4, "Revenue by Loyalty Tier")
        lt = df.groupby("loyalty_tier", observed=True)["total_sales"].sum().sort_values(ascending=False)
        bars4 = ax4.bar(lt.index, lt.values,
                        color=[P["a1"], P["a2"], P["a5"]][:len(lt)],
                        edgecolor="none", width=0.5)
        for bar in bars4:
            ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+lt.max()*0.02,
                     f"₹{bar.get_height()/1e6:.2f}M", ha="center",
                     fontsize=9.5, color=P["text"])
        ax4.yaxis.set_major_formatter(MILLION)
        ax4.set_ylabel("Sales Volume"); ax4.grid(axis="y")

    # ── Payment Method Donut ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    _card(ax5, "Payment Method Split")
    pay = df["payment_method"].value_counts()
    wedges_p, texts_p, autotexts_p = ax5.pie(
        pay.values, labels=pay.index,
        colors=CAT_COLORS[:len(pay)],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": P["bg"], "linewidth": 3, "width": 0.35},
        textprops={"color": P["text"], "fontsize": 9.5}
    )
    for at in autotexts_p:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")
    ax5.text(0, 0, f"{len(df):,}\nOrders", ha="center", va="center",
             fontsize=12, fontweight="bold", color=P["text"])

    # ── Avg Order Value by Category ───────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    _card(ax6, "Avg Order Value by Category")
    avg_cat = (df.groupby("product_category", observed=True)["total_sales"]
               .mean().sort_values(ascending=False))
    bars6 = ax6.barh(avg_cat.index, avg_cat.values,
                     color=CAT_COLORS[:len(avg_cat)], edgecolor="none", height=0.55)
    for bar, val in zip(bars6, avg_cat.values):
        ax6.text(val + avg_cat.max()*0.02, bar.get_y()+bar.get_height()/2,
                 f"₹{val:,.0f}", va="center", fontsize=9.5, color=P["text"])
    ax6.xaxis.set_major_formatter(CURRENCY)
    ax6.set_xlabel("Avg Sales"); ax6.grid(axis="x")

    _save(fig, out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD 2 — DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
def dashboard_distributions(df: pd.DataFrame, out_path: str):
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor=P["bg"])
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.96,
                        hspace=0.46, wspace=0.34)
    _title(fig, "📈  Distribution & Demographics Dashboard",
           "Profiling client segment ages, order valuations, ratings spread, and transactional features distributions")
    ax = axes.flatten()

    # Age histogram + KDE overlay
    _card(ax[0], "Customer Age Distribution")
    age = df["age"].dropna()
    ax[0].hist(age, bins=25, color=P["a1"], edgecolor=P["surface"],
               linewidth=0.8, alpha=0.75, density=True, label="Hist")
    kde_x = np.linspace(age.min(), age.max(), 300)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(age)
    ax[0].plot(kde_x, kde(kde_x), color=P["a3"], lw=3, label="KDE Fit")
    ax[0].axvline(age.mean(),   color=P["a5"], ls="--", lw=1.8, label=f"Mean: {age.mean():.1f}")
    ax[0].axvline(age.median(), color=P["a4"], ls=":",  lw=1.8, label=f"Med: {age.median():.0f}")
    ax[0].set_xlabel("Age")
    ax[0].set_ylabel("Probability Density")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[0].grid(axis="y")

    # Unit Price Violin by Category
    _card(ax[1], "Unit Price Spans by Category")
    cat_order = (df.groupby("product_category", observed=True)["unit_price"]
                 .median().sort_values(ascending=False).index.tolist())
    sns.violinplot(data=df, x="product_category", y="unit_price",
                   order=cat_order,
                   palette=CAT_COLORS[:len(cat_order)],
                   ax=ax[1], inner="quartile", linewidth=1.2)
    ax[1].set_xlabel("Category"); ax[1].set_ylabel("Unit Price (₹)")
    ax[1].yaxis.set_major_formatter(CURRENCY)
    plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right", fontsize=8.5)

    # Rating histogram (colour-coded by rating value)
    _card(ax[2], "Customer Rating Volumes")
    rating_colors = {1: P["a3"], 2: P["a5"], 3: P["a2"], 4: P["a1"], 5: P["a4"]}
    for r in range(1, 6):
        mask = df["customer_rating"].round().astype("Int64") == r
        ax[2].bar(r, mask.sum(), color=rating_colors[r],
                  edgecolor="none", width=0.55)
    ax[2].set_xlabel("Rating (⭐)"); ax[2].set_ylabel("Number of Orders")
    ax[2].set_xticks([1,2,3,4,5]); ax[2].grid(axis="y")

    # Discount distribution
    _card(ax[3], "Discount Rate Occurrences")
    disc = df["discount"] * 100
    ax[3].hist(disc, bins=15, color=P["a2"], edgecolor=P["surface"], alpha=0.85, width=2.0)
    ax[3].set_xlabel("Discount (%)"); ax[3].set_ylabel("Count")
    ax[3].grid(axis="y")

    # Total sales distribution
    _card(ax[4], "Order Valuation Volumes (total_sales)")
    ax[4].hist(df["total_sales"], bins=50, color=P["a5"],
               edgecolor=P["surface"], lw=0.5, alpha=0.85)
    ax[4].set_xlabel("Order Value (₹)"); ax[4].set_ylabel("Count")
    ax[4].xaxis.set_major_formatter(CURRENCY)
    plt.setp(ax[4].get_xticklabels(), rotation=30, ha="right", fontsize=8.5)
    ax[4].grid(axis="y")

    # Gender split donut (Hollow Donut)
    _card(ax[5], "Customer Gender Breakdown")
    gc = df["gender"].value_counts()
    wedges_g, texts_g, autotexts_g = ax[5].pie(
        gc.values, labels=gc.index,
        colors=[P["a1"], P["a3"]],
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": P["bg"], "linewidth": 3, "width": 0.35},
        textprops={"color": P["text"], "fontsize": 10}
    )
    for at in autotexts_g:
        at.set_fontsize(9); at.set_fontweight("bold")
    ax[5].text(0, 0, f"n={len(df):,}\nRegistered", ha="center", va="center",
               fontsize=11, fontweight="bold", color=P["text"])

    _save(fig, out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD 3 — CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
def dashboard_correlations(df: pd.DataFrame, out_path: str):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), facecolor=P["bg"])
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.96,
                        hspace=0.44, wspace=0.32)
    _title(fig, "🔗  Correlation & Feature Interaction Dashboard",
           "Exploring correlations, regression trends, and patterns of key categorical indicators")
    ax = axes.flatten()

    # 1. Correlation heatmap
    _card(ax[0], "Variables Correlation Matrix")
    num_cols = [c for c in ["age","quantity","unit_price","discount",
                             "total_sales","customer_rating","shipping_days"]
                if c in df.columns]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=2.5, linecolor=P["card"],
                ax=ax[0],
                annot_kws={"size": 10, "color": P["text"]},
                cbar_kws={"shrink": 0.75})
    ax[0].tick_params(labelsize=9)

    # 2. Discount vs Total Sales scatter + regression + R²
    _card(ax[1], "Discount Rate impact on Total Sales")
    sample = df.sample(min(800, len(df)), random_state=42)
    sc = ax[1].scatter(sample["discount"]*100, sample["total_sales"],
                       c=sample["customer_rating"], cmap="plasma",
                       alpha=0.65, s=28, edgecolors="none")
    m, b = np.polyfit(sample["discount"]*100, sample["total_sales"], 1)
    x_line = np.linspace(0, 25, 100)
    ax[1].plot(x_line, m*x_line + b, color=P["a3"], lw=2.5, ls="--",
               label=f"Trend (Slope: {m:.0f})")
    
    ss_res = ((sample["total_sales"] - (m*sample["discount"]*100+b))**2).sum()
    ss_tot = ((sample["total_sales"] - sample["total_sales"].mean())**2).sum()
    r2_val = 1 - ss_res/ss_tot
    ax[1].text(0.05, 0.95, f"$R^2$ Score: {r2_val:.3f}", ha="left", va="top",
               transform=ax[1].transAxes, fontsize=11,
               color=P["a1"], fontweight="bold")
    
    cbar = plt.colorbar(sc, ax=ax[1])
    cbar.set_label("Rating ⭐", color=P["sub"], fontsize=9.5)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=P["sub"])
    ax[1].set_xlabel("Discount (%)"); ax[1].set_ylabel("Order Value (₹)")
    ax[1].yaxis.set_major_formatter(CURRENCY)
    ax[1].legend(fontsize=9, loc="upper right"); ax[1].grid()

    # 3. Avg Sales by Payment Method
    _card(ax[2], "Average Order Value by Payment Method")
    pay_avg = (df.groupby("payment_method", observed=True)["total_sales"]
               .mean().sort_values())
    bars = ax[2].barh(pay_avg.index, pay_avg.values,
                      color=CAT_COLORS[:len(pay_avg)], edgecolor="none", height=0.5)
    for bar, val in zip(bars, pay_avg.values):
        ax[2].text(val + pay_avg.max()*0.02, bar.get_y()+bar.get_height()/2,
                   f"₹{val:,.0f}", va="center", fontsize=9.5, color=P["text"])
    ax[2].xaxis.set_major_formatter(CURRENCY)
    ax[2].set_xlabel("Avg Sales (₹)"); ax[2].grid(axis="x")

    # 4. Rating boxplot by Category
    _card(ax[3], "Customer Rating Spreads by Category")
    cat_order = (df.groupby("product_category", observed=True)["customer_rating"]
                 .mean().sort_values(ascending=False).index.tolist())
    sns.boxplot(data=df, x="product_category", y="customer_rating",
                order=cat_order, palette=CAT_COLORS, ax=ax[3],
                linewidth=1.2, width=0.55,
                flierprops={"marker":"o","markersize":3.5,"markerfacecolor":P["sub"], "markeredgecolor":"none"})
    ax[3].set_xlabel("Category"); ax[3].set_ylabel("Rating (1-5)")
    ax[3].set_ylim(0.8, 5.2)
    plt.setp(ax[3].get_xticklabels(), rotation=30, ha="right", fontsize=8.5)
    ax[3].grid(axis="y")

    _save(fig, out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD 4 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
def dashboard_trends(df: pd.DataFrame, out_path: str):
    _style()
    fig = plt.figure(figsize=(22, 15), facecolor=P["bg"])
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.05, right=0.96,
                        hspace=0.46, wspace=0.34)
    _title(fig, "📅  Temporal Trends & Product Leadership Dashboard",
           "Comparing quarterly performance metrics, category breakdowns, and transaction method usage summaries")
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Quarterly Revenue (grouped bars + YoY growth %)
    ax1 = fig.add_subplot(gs[0, :2])
    _card(ax1, "Quarterly Revenue Comparison (2023 vs 2024)")
    qtr    = (df.groupby(["year","quarter"], observed=True)["total_sales"]
              .sum().reset_index())
    years  = sorted(qtr["year"].unique())
    quarts = [1,2,3,4]
    x      = np.arange(len(quarts))
    width  = 0.32
    colors_yr = [P["a1"], P["a2"]]
    val_by_yr = {}
    for i, yr in enumerate(years):
        vals = [qtr[(qtr["year"]==yr) & (qtr["quarter"]==q)]["total_sales"].sum()
                for q in quarts]
        val_by_yr[yr] = vals
        bars = ax1.bar(x + i*width - width/2, vals, width,
                       color=colors_yr[i], edgecolor="none",
                       label=str(yr), alpha=0.9)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5000,
                     f"₹{val/1e6:.1f}M", ha="center", fontsize=8.5, color=P["text"])
    
    # YoY growth labels
    if len(years) >= 2:
        for qi, q in enumerate(quarts):
            v0 = val_by_yr[years[0]][qi]
            v1 = val_by_yr[years[1]][qi]
            if v0 > 0:
                pct = (v1-v0)/v0*100
                col = P["a4"] if pct >= 0 else P["a3"]
                ax1.text(x[qi], max(v0,v1)*1.08, f"{pct:+.1f}% YoY",
                         ha="center", fontsize=9.5, color=col, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(["Q1","Q2","Q3","Q4"])
    ax1.yaxis.set_major_formatter(CURRENCY)
    ax1.set_ylabel("Sales Value (₹)")
    ax1.legend(title="Year", fontsize=9, loc="upper left"); ax1.grid(axis="y")

    # Stacked Region × Category
    ax2 = fig.add_subplot(gs[0, 2])
    _card(ax2, "Category Product Split by Region")
    pivot = (df.groupby(["region","product_category"], observed=True)["total_sales"]
             .sum().unstack(fill_value=0))
    pivot.plot(kind="bar", stacked=True, ax=ax2,
               color=CAT_COLORS[:len(pivot.columns)],
               edgecolor=P["bg"], lw=0.8, width=0.55)
    ax2.yaxis.set_major_formatter(CURRENCY)
    ax2.set_xlabel("Region"); ax2.set_ylabel("Sales Volume (₹)")
    ax2.legend(title="Category", fontsize=8.5, title_fontsize=8.5,
               loc="upper right", framealpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax2.grid(axis="y")

    # Top 10 Products
    ax3 = fig.add_subplot(gs[1, :2])
    _card(ax3, "Top 10 Performing Products (Revenue)")
    top10 = (df.groupby("product_name")["total_sales"]
             .sum().sort_values(ascending=True).tail(10))
    colors_top = plt.cm.plasma(np.linspace(0.35, 0.85, 10))
    bars3 = ax3.barh(top10.index, top10.values,
                     color=colors_top, edgecolor="none", height=0.55)
    for bar, val in zip(bars3, top10.values):
        ax3.text(val + top10.max()*0.015, bar.get_y()+bar.get_height()/2,
                 f"₹{val/1e6:.2f}M", va="center", fontsize=9.5, color=P["text"])
    ax3.xaxis.set_major_formatter(CURRENCY)
    ax3.set_xlabel("Total Sales (₹)"); ax3.grid(axis="x")

    # Payment mix donut (Upgraded from generic pie)
    ax4 = fig.add_subplot(gs[1, 2])
    _card(ax4, "Payment Methods Preference Mix")
    pc = df["payment_method"].value_counts()
    wedges_pay, texts_pay, autotexts_pay = ax4.pie(
        pc.values, labels=pc.index,
        colors=CAT_COLORS[:len(pc)],
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": P["bg"], "linewidth": 3, "width": 0.35},
        textprops={"color": P["text"], "fontsize": 9.5}
    )
    for at in autotexts_pay:
        at.set_fontsize(8.5); at.set_fontweight("bold")
    ax4.text(0, 0, f"n={len(df):,}\nOrders", ha="center", va="center",
             fontsize=11, fontweight="bold", color=P["text"])

    _save(fig, out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD 5 — ML INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def dashboard_ml_insights(df: pd.DataFrame, ml_results: dict,
                           analysis_results: dict, out_path: str):
    _style()
    fig = plt.figure(figsize=(22, 15), facecolor=P["bg"])
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.96,
                        hspace=0.46, wspace=0.34)
    metrics = ml_results.get("metrics", {})
    _title(
        fig, "🤖  Machine Learning Diagnostics & Product ABC segmentation",
        f"Random Forest Model  |  $R^2$ Score: {metrics.get('R2','N/A')}  "
        f"MAE: ₹{metrics.get('MAE',0):,.0f}  "
        f"RMSE: ₹{metrics.get('RMSE',0):,.0f}"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # 1. Feature Importance
    ax1 = fig.add_subplot(gs[0, 0])
    _card(ax1, "Random Forest Feature Importance")
    fi = ml_results.get("feature_importances", {})
    top_n = 10
    items = list(fi.items())[:top_n]
    names, vals = zip(*items) if items else ([], [])
    
    y_pos = np.arange(len(names))
    bars = ax1.barh(y_pos, vals, color=P["a1"], edgecolor="none", height=0.55)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    ax1.set_xlabel("Relative Importance Score")
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + max(vals)*0.015, bar.get_y()+bar.get_height()/2,
                 f"{w:.3f}", va="center", fontsize=9, color=P["text"])
    ax1.grid(axis="x")

    # 2. Actual vs Predicted Scatter
    ax2 = fig.add_subplot(gs[0, 1])
    _card(ax2, "Model Actual vs. Predicted daily revenue")
    eval_df = ml_results.get("evaluation_df")
    if eval_df is not None and not eval_df.empty:
        ax2.scatter(eval_df["actual"], eval_df["predicted"],
                    color=P["a2"], alpha=0.6, s=25, edgecolor="none", label="Test days")
        lo = min(eval_df["actual"].min(), eval_df["predicted"].min())
        hi = max(eval_df["actual"].max(), eval_df["predicted"].max())
        ax2.plot([lo, hi], [lo, hi], color=P["a3"], ls="--", lw=2, label="Perfect Fit (y=x)")
        ax2.set_xlabel("Actual Sales (₹)"); ax2.set_ylabel("Predicted Sales (₹)")
        ax2.xaxis.set_major_formatter(CURRENCY)
        ax2.yaxis.set_major_formatter(CURRENCY)
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid()

    # 3. ABC Product Segment Volume
    ax3 = fig.add_subplot(gs[0, 2])
    _card(ax3, "Revenue Split by Product ABC Segment")
    abc = analysis_results.get("abc")
    if abc is not None and not abc.empty:
        abc_counts = abc.groupby("abc_tier")["revenue"].sum()
        bars_abc = ax3.bar(abc_counts.index, abc_counts.values,
                           color=[P["a4"], P["a1"], P["a3"]][:len(abc_counts)],
                           edgecolor="none", width=0.5)
        for bar in bars_abc:
            h = bar.get_height()
            ax3.text(bar.get_x()+bar.get_width()/2, h + abc_counts.max()*0.02,
                     f"₹{h/1e6:.1f}M", ha="center", fontsize=9.5, color=P["text"])
        ax3.set_ylabel("Accumulated revenue (₹)")
        ax3.yaxis.set_major_formatter(MILLION)
        ax3.grid(axis="y")

    # 4. Revenue forecast (next 3 months)
    ax4 = fig.add_subplot(gs[1, :])
    _card(ax4, "3-Month Sales Revenue Forecast projection (Daily Timeline)")
    
    daily_hist = ml_results.get("daily_historical")
    forecast_df = ml_results.get("daily_forecast")
    
    if daily_hist is not None and forecast_df is not None:
        # Show recent 90 days of history for comparison
        recent_hist = daily_hist.tail(90)
        ax4.plot(recent_hist["order_date"], recent_hist["total_sales"],
                 color=P["sub"], lw=2, label="Recent History (Actual)")
        
        # Plot forecast
        ax4.plot(forecast_df["order_date"], forecast_df["predicted_sales"],
                 color=P["a1"], lw=2.5, label="Forecast Projection")
        
        # Fill confidence interval
        ax4.fill_between(forecast_df["order_date"],
                         forecast_df["lower"], forecast_df["upper"],
                         color=P["a1"], alpha=0.15, label="95% Confidence Interval")
        
        ax4.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
        ax4.yaxis.set_major_formatter(CURRENCY)
        ax4.set_ylabel("Daily Sales Revenue")
        ax4.legend(fontsize=9.5, loc="upper left")
        ax4.grid(axis="y")
        
    _save(fig, out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def generate_all_dashboards(
    df:               pd.DataFrame,
    output_dir:       str,
    ml_results:       Optional[dict] = None,
    analysis_results: Optional[dict] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("\n  Generating dashboards …")

    dashboard_overview(df,
        os.path.join(output_dir, "dashboard_1_overview.png"))
    dashboard_distributions(df,
        os.path.join(output_dir, "dashboard_2_distributions.png"))
    dashboard_correlations(df,
        os.path.join(output_dir, "dashboard_3_correlations.png"))
    dashboard_trends(df,
        os.path.join(output_dir, "dashboard_4_trends.png"))

    if ml_results and analysis_results is not None:
        dashboard_ml_insights(df, ml_results, analysis_results,
            os.path.join(output_dir, "dashboard_5_ml_insights.png"))

    print(f"\n  All dashboards saved → {os.path.abspath(output_dir)}\n")
