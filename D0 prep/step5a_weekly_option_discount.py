"""
TCNS D0 - Step 5a: Discount % for step5 weekly sales.

Formula (same as Tasva):
  Unit_GSV = Sales_GSV / Sales_Qty
  Discount_Pct = (MRP - Unit_GSV) / MRP * 100

Also builds Week x Option Code rollup:
  Discount_Pct = (sum(MRP * Qty) - sum(GSV)) / sum(MRP * Qty) * 100
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
from paths import out_dir, write_split

OUT = out_dir("step5_skd_3n_filter")
SALES_IN = OUT / "sales_weekly_8ag_skd_3n_filtered.parquet"


def add_row_discount(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    qty = pd.to_numeric(out["Sales_Qty"], errors="coerce")
    gsv = pd.to_numeric(out["Sales_GSV"], errors="coerce")
    mrp = pd.to_numeric(out["MRP"], errors="coerce")
    mrp_c = pd.to_numeric(out.get("MRP_corrected", mrp), errors="coerce")

    out["Unit_GSV"] = np.where(qty > 0, gsv / qty, np.nan)
    out["Discount_Pct"] = np.where(
        (mrp > 0) & qty.notna() & (qty > 0) & gsv.notna(),
        (mrp - out["Unit_GSV"]) / mrp * 100.0,
        np.nan,
    )
    out["Discount_Pct_vs_MRP_corrected"] = np.where(
        (mrp_c > 0) & qty.notna() & (qty > 0) & gsv.notna(),
        (mrp_c - out["Unit_GSV"]) / mrp_c * 100.0,
        np.nan,
    )
    return out


def week_option_discount(df: pd.DataFrame) -> pd.DataFrame:
    """One row per Date (week) x Option Code."""
    t = df.copy()
    t["Date"] = pd.to_datetime(t["Date"])
    t["Sales_Qty"] = pd.to_numeric(t["Sales_Qty"], errors="coerce")
    t["Sales_GSV"] = pd.to_numeric(t["Sales_GSV"], errors="coerce")
    t["Sales_NSV"] = pd.to_numeric(t["Sales_NSV"], errors="coerce")
    t["MRP"] = pd.to_numeric(t["MRP"], errors="coerce")
    t["MRP_corrected"] = pd.to_numeric(t.get("MRP_corrected", t["MRP"]), errors="coerce")
    t["mrp_x_qty"] = t["MRP"] * t["Sales_Qty"]
    t["mrp_c_x_qty"] = t["MRP_corrected"] * t["Sales_Qty"]

    g = (
        t.groupby(["Date", "Option Code"], as_index=False)
        .agg(
            AG=("AG", "first"),
            scope_category=("scope_category", "first"),
            Sales_Qty=("Sales_Qty", "sum"),
            Sales_GSV=("Sales_GSV", "sum"),
            Sales_NSV=("Sales_NSV", "sum"),
            mrp_x_qty=("mrp_x_qty", "sum"),
            mrp_c_x_qty=("mrp_c_x_qty", "sum"),
            unique_sites=("Site_Code", "nunique"),
            n_rows=("Option Code", "size"),
            MRP_mode=("MRP", lambda s: s.value_counts().index[0] if len(s) else pd.NA),
            MRP_corrected_mode=(
                "MRP_corrected",
                lambda s: s.value_counts().index[0] if len(s) else pd.NA,
            ),
        )
    )
    g["Discount_Pct"] = np.where(
        g["mrp_x_qty"] > 0,
        (g["mrp_x_qty"] - g["Sales_GSV"]) / g["mrp_x_qty"] * 100.0,
        np.nan,
    )
    g["Discount_Pct_vs_MRP_corrected"] = np.where(
        g["mrp_c_x_qty"] > 0,
        (g["mrp_c_x_qty"] - g["Sales_GSV"]) / g["mrp_c_x_qty"] * 100.0,
        np.nan,
    )
    g["Unit_GSV"] = np.where(g["Sales_Qty"] > 0, g["Sales_GSV"] / g["Sales_Qty"], np.nan)
    g = g.drop(columns=["mrp_x_qty", "mrp_c_x_qty"])
    g = g.sort_values(["Date", "Option Code"]).reset_index(drop=True)
    return g


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Input:", SALES_IN.exists(), SALES_IN)
    if not SALES_IN.exists():
        raise FileNotFoundError(f"Run step5 first. Missing: {SALES_IN}")

    print("\n1) Load step5 filtered sales...")
    df = pd.read_parquet(SALES_IN)
    print(f"  rows {len(df):,} | options {df['Option Code'].nunique():,}")

    print("\n2) Add row-level Discount_Pct (MRP vs Unit_GSV)...")
    with_disc = add_row_discount(df)
    usable = with_disc["Discount_Pct"].notna().sum()
    print(
        f"  rows with Discount_Pct: {usable:,} | "
        f"mean {with_disc['Discount_Pct'].mean():.2f}% | "
        f"median {with_disc['Discount_Pct'].median():.2f}%"
    )

    out_rows = OUT / "sales_weekly_8ag_with_discount.parquet"
    with_disc.to_parquet(out_rows, index=False)
    with_disc.head(100_000).to_csv(OUT / "sales_weekly_8ag_with_discount_sample.csv", index=False)
    print(f"  wrote {out_rows}")

    print("\n3) Week x Option Code discount...")
    wo = week_option_discount(with_disc)
    print(f"  week x option rows: {len(wo):,}")
    print(
        f"  mean Discount_Pct: {wo['Discount_Pct'].mean():.2f}% | "
        f"median: {wo['Discount_Pct'].median():.2f}%"
    )

    wo_pq = OUT / "discount_by_week_option.parquet"
    wo_csv = OUT / "discount_by_week_option.csv"
    wo.to_parquet(wo_pq, index=False)
    wo.to_csv(wo_csv, index=False)
    print("\nSplitting step 5a into the four scope folders...")
    write_split(with_disc, "step5a_discount", "sales_weekly_with_discount.parquet")
    write_split(wo, "step5a_discount", "discount_by_week_option.parquet")
    print(f"  wrote {wo_pq}")
    print(f"  wrote {wo_csv}")

    # light summary by AG
    by_ag = (
        wo.groupby(["AG", "scope_category"], as_index=False)
        .agg(
            week_option_rows=("Option Code", "size"),
            unique_options=("Option Code", "nunique"),
            unique_weeks=("Date", "nunique"),
            Sales_Qty=("Sales_Qty", "sum"),
            Sales_GSV=("Sales_GSV", "sum"),
            avg_Discount_Pct=("Discount_Pct", "mean"),
            median_Discount_Pct=("Discount_Pct", "median"),
        )
    )
    by_ag.to_csv(OUT / "discount_summary_by_ag.csv", index=False)

    overall = {
        "formula": "Discount_Pct = (MRP - Sales_GSV/Sales_Qty) / MRP * 100",
        "week_option_formula": "Discount_Pct = (sum(MRP*Qty) - sum(GSV)) / sum(MRP*Qty) * 100",
        "rows_with_discount": int(usable),
        "week_option_rows": len(wo),
        "mean_Discount_Pct_row": round(float(with_disc["Discount_Pct"].mean()), 2),
        "median_Discount_Pct_row": round(float(with_disc["Discount_Pct"].median()), 2),
        "mean_Discount_Pct_week_option": round(float(wo["Discount_Pct"].mean()), 2),
        "median_Discount_Pct_week_option": round(float(wo["Discount_Pct"].median()), 2),
    }
    pd.DataFrame([overall]).to_csv(OUT / "step5a_discount_summary.csv", index=False)

    print("\n=== BY AG (week x option) ===")
    print(by_ag.to_string(index=False))
    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print("DONE - discount by week x option complete.")
    return overall


if __name__ == "__main__":
    run()
