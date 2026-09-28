"""
TCNS D0 — Step 2: Daily sales → weekly (aligned to stock Mondays).

1. Load daily sales (merged parquet or Sale Data CSVs)
2. Remove Qty <= 0 (negatives and zeros), then each daily row with discount > 60%
   (Discount = (MRP_corrected - GSV/Qty) / MRP_corrected * 100,
    MRP_corrected snaps that day's MRP to the nearest price ending in 99)
3. Build week calendar from Step-1 stock Mondays; fill pre-stock gap with ISO Mondays
4. Map each sales day forward to Week_End_Date (Monday)
5. Aggregate weekly at Site × Option Code grain
6. Keep Date = Week_End_Date (same as stock Date) + min/max sale dates
7. Validate: weekly totals == filtered daily totals per Option Code
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
from paths import SALES_DIR, SALES_PQ, out_dir

OUT = out_dir("step2_weekly_sales")
STOCK_CAL = out_dir("step1_stock_formatted") / "stock_week_calendar.csv"

TOL = 1e-6
DISCOUNT_THRESHOLD = 60.0


def load_daily_sales() -> pd.DataFrame:
    if SALES_PQ.exists() and SALES_PQ.stat().st_size > 1000:
        print(f"Loading sales parquet: {SALES_PQ}")
        sales = pd.read_parquet(SALES_PQ)
    else:
        print(f"Loading sales CSVs from: {SALES_DIR}")
        parts = []
        for f in sorted(SALES_DIR.glob("*.csv")):
            df = pd.read_csv(f, low_memory=False)
            df["source_file"] = f.name
            parts.append(df)
        sales = pd.concat(parts, ignore_index=True)

    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce", format="mixed").dt.normalize()
    sales["Option Code"] = (
        sales.get("Style_code", sales.get("Option Code"))
        .astype("string")
        .str.strip()
        .str.upper()
    )
    sales["Site_Code"] = sales["Site_Code"].astype("string").str.strip().str.upper()
    sales["Zone"] = sales["Zone"].astype("string").str.strip()
    if "Category" in sales.columns:
        sales["Category"] = sales["Category"].astype("string").str.strip().str.upper()
    if "subcategory" in sales.columns:
        sales["subcategory"] = sales["subcategory"].astype("string").str.strip().str.upper()

    for c, src in [
        ("Sales_Qty", "Qty"),
        ("Sales_NSV", "nsv"),
        ("Sales_GSV", "gsv"),
        ("MRP", "mrp"),
        ("tax", "tax"),
    ]:
        if src in sales.columns:
            sales[c] = pd.to_numeric(sales[src], errors="coerce")
        elif c in sales.columns:
            sales[c] = pd.to_numeric(sales[c], errors="coerce")

    return sales


def drop_non_positive_qty(sales: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Remove Qty < 0 and Qty == 0."""
    before = len(sales)
    qty = sales["Sales_Qty"]
    mask_neg = qty < 0
    mask_zero = qty == 0
    mask_drop = mask_neg | mask_zero | qty.isna()
    removed = sales.loc[mask_drop].copy()
    kept = sales.loc[~mask_drop].copy()
    stats = {
        "rows_before": before,
        "rows_removed_qty_lt_0": int(mask_neg.sum()),
        "rows_removed_qty_eq_0": int(mask_zero.sum()),
        "rows_removed_qty_null": int(qty.isna().sum()),
        "rows_removed_total": int(mask_drop.sum()),
        "rows_kept": len(kept),
        "qty_removed": float(removed["Sales_Qty"].sum(skipna=True)),
        "nsv_removed": float(removed["Sales_NSV"].sum(skipna=True)),
        "qty_kept": float(kept["Sales_Qty"].sum(skipna=True)),
        "nsv_kept": float(kept["Sales_NSV"].sum(skipna=True)),
    }
    print(
        f"  dropped Qty<=0 / null: {stats['rows_removed_total']:,} "
        f"(neg={stats['rows_removed_qty_lt_0']:,}, zero={stats['rows_removed_qty_eq_0']:,}, "
        f"null={stats['rows_removed_qty_null']:,}) | kept {stats['rows_kept']:,}"
    )
    return kept, removed, stats


def correct_mrp_to_x99(mrp: pd.Series) -> pd.Series:
    """Snap to nearest ...99 with a 100 gap. Floor at 99."""
    x = pd.to_numeric(mrp, errors="coerce")
    out = np.round((x - 99.0) / 100.0) * 100.0 + 99.0
    out = out.where(x.notna(), np.nan)
    return out.clip(lower=99.0)


def daily_discount(sales: pd.DataFrame) -> pd.Series:
    """Discount of one store-style-day versus that day's corrected MRP."""
    qty = sales["Sales_Qty"]
    gsv = sales["Sales_GSV"]
    mrp_c = correct_mrp_to_x99(sales["MRP"])
    unit_gsv = gsv / qty.where(qty > 0)
    return ((mrp_c - unit_gsv) / mrp_c.where(mrp_c > 0)) * 100.0


def drop_high_discount(
    sales: pd.DataFrame, threshold: float = DISCOUNT_THRESHOLD
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Remove daily rows with discount > threshold. A day with no discount is kept."""
    before = len(sales)
    disc = daily_discount(sales)
    mask_drop = disc > threshold
    removed = sales.loc[mask_drop].copy()
    removed["Discount_Pct_vs_MRP_corrected"] = disc[mask_drop]
    kept = sales.loc[~mask_drop].copy()
    qty_before = float(sales["Sales_Qty"].sum(skipna=True))
    stats = {
        "threshold": threshold,
        "rows_before": before,
        "rows_removed_discount_gt_threshold": int(mask_drop.sum()),
        "rows_discount_nan_kept": int(disc.isna().sum()),
        "rows_kept": len(kept),
        "options_before": int(sales["Option Code"].nunique()),
        "options_kept": int(kept["Option Code"].nunique()),
        "qty_before": qty_before,
        "qty_removed": float(removed["Sales_Qty"].sum(skipna=True)),
        "qty_removed_pct": round(
            100.0 * float(removed["Sales_Qty"].sum(skipna=True)) / qty_before, 2
        )
        if qty_before
        else 0.0,
        "gsv_removed": float(removed["Sales_GSV"].sum(skipna=True)),
        "nsv_removed": float(removed["Sales_NSV"].sum(skipna=True)),
        "qty_kept": float(kept["Sales_Qty"].sum(skipna=True)),
        "gsv_kept": float(kept["Sales_GSV"].sum(skipna=True)),
        "nsv_kept": float(kept["Sales_NSV"].sum(skipna=True)),
        "qty_removed_pct_if_cut_were_50": round(
            100.0 * float(sales.loc[disc > 50, "Sales_Qty"].sum(skipna=True)) / qty_before, 2
        )
        if qty_before
        else 0.0,
    }
    print(
        f"  dropped Discount>{threshold:g}%: {stats['rows_removed_discount_gt_threshold']:,} "
        f"(NaN discount kept={stats['rows_discount_nan_kept']:,}) | kept {stats['rows_kept']:,} | "
        f"units removed {stats['qty_removed_pct']}%"
    )
    return kept, removed, stats


def load_stock_week_calendar(path: Path = STOCK_CAL) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Stock week calendar not found: {path}\nRun step1_stock_formatted first."
        )
    cal = pd.read_csv(path, parse_dates=["Date", "Week_Start"])
    cal = cal.rename(columns={"Date": "Week_End_Date"})
    cal["Week_End_Date"] = pd.to_datetime(cal["Week_End_Date"]).dt.normalize()
    cal = (
        cal[["Week_End_Date", "Week", "Month", "Year", "Week_Start"]]
        .drop_duplicates("Week_End_Date")
        .sort_values("Week_End_Date")
        .reset_index(drop=True)
    )
    cal["week_source"] = "stock"
    return cal


def extend_calendar_for_sales(
    stock_weeks: pd.DataFrame, sales_dates: pd.Series
) -> pd.DataFrame:
    """
    Stock Mondays cover mid/late period; sales starts earlier.
    Add ISO Mondays from first sales date up to (but not including) first stock Monday,
    so early sales are not dumped into one stock week.
    """
    smin = pd.to_datetime(sales_dates.min()).normalize()
    smax = pd.to_datetime(sales_dates.max()).normalize()
    first_stock = stock_weeks["Week_End_Date"].min()
    last_stock = stock_weeks["Week_End_Date"].max()

    parts = [stock_weeks.copy()]

    if smin < first_stock:
        # Mondays from sales start week through the Monday before first stock
        pre = pd.date_range(
            start=smin - pd.Timedelta(days=int(smin.dayofweek)),  # Monday on/before smin
            end=first_stock - pd.Timedelta(days=7),
            freq="W-MON",
        )
        if len(pre):
            pre_df = pd.DataFrame({"Week_End_Date": pre})
            pre_df["Week"] = pre_df["Week_End_Date"].dt.isocalendar().week.astype(int)
            pre_df["Month"] = pre_df["Week_End_Date"].dt.month
            pre_df["Year"] = pre_df["Week_End_Date"].dt.year
            pre_df["Week_Start"] = pre_df["Week_End_Date"]
            pre_df["week_source"] = "iso_pre_stock"
            parts.append(pre_df)
            print(f"  added {len(pre_df)} pre-stock ISO Mondays: {pre_df['Week_End_Date'].min().date()} -> {pre_df['Week_End_Date'].max().date()}")

    if smax > last_stock:
        post = pd.date_range(
            start=last_stock + pd.Timedelta(days=7),
            end=smax + pd.Timedelta(days=(7 - int(smax.dayofweek)) % 7),
            freq="W-MON",
        )
        # Also include Monday of the week containing smax
        last_mon = smax - pd.Timedelta(days=int(smax.dayofweek))
        if last_mon > last_stock and (len(post) == 0 or last_mon not in set(post)):
            post = post.union(pd.DatetimeIndex([last_mon])).sort_values()
        if len(post):
            post_df = pd.DataFrame({"Week_End_Date": post})
            post_df["Week"] = post_df["Week_End_Date"].dt.isocalendar().week.astype(int)
            post_df["Month"] = post_df["Week_End_Date"].dt.month
            post_df["Year"] = post_df["Week_End_Date"].dt.year
            post_df["Week_Start"] = post_df["Week_End_Date"]
            post_df["week_source"] = "iso_post_stock"
            parts.append(post_df)
            print(f"  added {len(post_df)} post-stock ISO Mondays")

    weeks = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("Week_End_Date")
        .sort_values("Week_End_Date")
        .reset_index(drop=True)
    )
    return weeks


def map_sales_to_weeks(sales: pd.DataFrame, weeks: pd.DataFrame) -> pd.DataFrame:
    """Each sales day → next Week_End_Date (Monday) on/after that day."""
    s = sales.sort_values("Date").copy()
    w = weeks.sort_values("Week_End_Date")[
        ["Week_End_Date", "Week", "Month", "Year", "Week_Start", "week_source"]
    ].copy()
    mapped = pd.merge_asof(
        s,
        w,
        left_on="Date",
        right_on="Week_End_Date",
        direction="forward",
    )
    return mapped


def aggregate_weekly(mapped: pd.DataFrame) -> pd.DataFrame:
    """
    Daily → weekly at Site × Option Code.
    Date column kept as Week_End_Date (same Monday anchor as stock Date).
    Original sale day range kept as min_sale_date / max_sale_date.
    """
    keys = [
        "Week_End_Date",
        "Week",
        "Month",
        "Year",
        "Week_Start",
        "week_source",
        "Zone",
        "Site_Code",
        "Option Code",
        "Category",
        "subcategory",
    ]
    keys = [k for k in keys if k in mapped.columns]

    agg = {
        "Sales_Qty": ("Sales_Qty", "sum"),
        "Sales_NSV": ("Sales_NSV", "sum"),
        "Sales_GSV": ("Sales_GSV", "sum"),
        "MRP": ("MRP", "median"),
        "n_daily_rows": ("Date", "size"),
        "min_sale_date": ("Date", "min"),
        "max_sale_date": ("Date", "max"),
    }
    if "tax" in mapped.columns:
        agg["tax"] = ("tax", "sum")

    g = (
        mapped.dropna(subset=["Week_End_Date"])
        .groupby(keys, dropna=False, as_index=False)
        .agg(**agg)
    )
    g["Date"] = g["Week_End_Date"]
    cols = [
        "Date",
        "Week_End_Date",
        "Week_Start",
        "Year",
        "Month",
        "Week",
        "week_source",
        "Zone",
        "Site_Code",
        "Option Code",
        "Category",
        "subcategory",
        "Sales_Qty",
        "Sales_NSV",
        "Sales_GSV",
        "MRP",
        "tax",
        "n_daily_rows",
        "min_sale_date",
        "max_sale_date",
    ]
    cols = [c for c in cols if c in g.columns]
    return (
        g[cols]
        .sort_values(["Date", "Site_Code", "Option Code"], kind="mergesort")
        .reset_index(drop=True)
    )


def option_totals(df: pd.DataFrame, qty: str, nsv: str, gsv: str, opt: str = "Option Code") -> pd.DataFrame:
    return (
        df.groupby(opt, dropna=False, as_index=False)
        .agg(qty=(qty, "sum"), nsv=(nsv, "sum"), gsv=(gsv, "sum"), rows=(opt, "size"))
        .sort_values(opt)
        .reset_index(drop=True)
    )


def validate(daily: pd.DataFrame, weekly: pd.DataFrame, mapped: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = option_totals(daily, "Sales_Qty", "Sales_NSV", "Sales_GSV").rename(
        columns={"qty": "daily_qty", "nsv": "daily_nsv", "gsv": "daily_gsv", "rows": "daily_rows"}
    )
    w = option_totals(weekly, "Sales_Qty", "Sales_NSV", "Sales_GSV").rename(
        columns={"qty": "weekly_qty", "nsv": "weekly_nsv", "gsv": "weekly_gsv", "rows": "weekly_rows"}
    )
    unmapped = mapped.loc[mapped["Week_End_Date"].isna()]
    u = option_totals(unmapped, "Sales_Qty", "Sales_NSV", "Sales_GSV").rename(
        columns={
            "qty": "unmapped_qty",
            "nsv": "unmapped_nsv",
            "gsv": "unmapped_gsv",
            "rows": "unmapped_rows",
        }
    )
    cmp = d.merge(w, on="Option Code", how="outer").merge(u, on="Option Code", how="left")
    for c in [
        "daily_qty",
        "daily_nsv",
        "daily_gsv",
        "daily_rows",
        "weekly_qty",
        "weekly_nsv",
        "weekly_gsv",
        "weekly_rows",
        "unmapped_qty",
        "unmapped_nsv",
        "unmapped_gsv",
        "unmapped_rows",
    ]:
        if c in cmp.columns:
            cmp[c] = cmp[c].fillna(0)

    cmp["qty_diff"] = cmp["daily_qty"] - cmp["weekly_qty"] - cmp["unmapped_qty"]
    cmp["nsv_diff"] = cmp["daily_nsv"] - cmp["weekly_nsv"] - cmp["unmapped_nsv"]
    cmp["gsv_diff"] = cmp["daily_gsv"] - cmp["weekly_gsv"] - cmp["unmapped_gsv"]
    cmp["ok"] = (
        cmp["qty_diff"].abs().le(TOL)
        & cmp["nsv_diff"].abs().le(TOL)
        & cmp["gsv_diff"].abs().le(TOL)
    )

    summary = {
        "daily_rows": len(daily),
        "weekly_rows": len(weekly),
        "unmapped_daily_rows": len(unmapped),
        "unique_options_daily": int(daily["Option Code"].nunique()),
        "unique_options_weekly": int(weekly["Option Code"].nunique()),
        "unique_sites_weekly": int(weekly["Site_Code"].nunique()),
        "weeks": int(weekly["Date"].nunique()),
        "date_min": str(weekly["Date"].min().date()) if len(weekly) else None,
        "date_max": str(weekly["Date"].max().date()) if len(weekly) else None,
        "total_qty_daily": float(daily["Sales_Qty"].sum()),
        "total_qty_weekly": float(weekly["Sales_Qty"].sum()),
        "total_nsv_daily": float(daily["Sales_NSV"].sum()),
        "total_nsv_weekly": float(weekly["Sales_NSV"].sum()),
        "options_with_mismatch": int((~cmp["ok"]).sum()),
        "all_ok": bool(cmp["ok"].all()),
    }
    return cmp, summary


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Output:", OUT)

    print("\n1) Load daily sales...")
    sales = load_daily_sales()
    print(f"  rows={len(sales):,}  options={sales['Option Code'].nunique()}  "
          f"dates={sales['Date'].min().date()} -> {sales['Date'].max().date()}")

    print("\n2) Remove Qty <= 0 (and null)...")
    sales, removed, drop_stats = drop_non_positive_qty(sales)
    pd.DataFrame([drop_stats]).to_csv(OUT / "daily_nonpositive_qty_drop_summary.csv", index=False)
    if len(removed):
        # sample of removed for audit (full can be large)
        removed.head(50_000).to_csv(OUT / "daily_sales_removed_qty_le_0_sample.csv", index=False)
        removed.groupby(
            removed["Sales_Qty"].apply(
                lambda x: "neg" if pd.notna(x) and x < 0 else ("zero" if x == 0 else "null")
            )
        ).agg(
            rows=("Sales_Qty", "size"),
            qty=("Sales_Qty", "sum"),
            nsv=("Sales_NSV", "sum"),
        ).rename_axis("reason").reset_index().to_csv(
            OUT / "daily_removed_by_reason.csv", index=False
        )

    print(f"\n2b) Remove daily rows with Discount > {DISCOUNT_THRESHOLD:g}% (vs that day's MRP)...")
    sales, removed_disc, disc_stats = drop_high_discount(sales)
    pd.DataFrame([disc_stats]).to_csv(OUT / "daily_discount_gt60_drop_summary.csv", index=False)
    if len(removed_disc):
        removed_disc.head(50_000).to_csv(
            OUT / "daily_sales_removed_discount_gt60_sample.csv", index=False
        )

    print("\n3) Stock week calendar + extend for full sales span...")
    stock_weeks = load_stock_week_calendar()
    print(f"  stock Mondays: {len(stock_weeks)}  "
          f"{stock_weeks['Week_End_Date'].min().date()} -> {stock_weeks['Week_End_Date'].max().date()}")
    weeks = extend_calendar_for_sales(stock_weeks, sales["Date"])
    weeks.to_csv(OUT / "sales_week_calendar.csv", index=False)
    print(f"  full week calendar: {len(weeks)} Mondays")

    print("\n4) Map sales days -> Week_End_Date...")
    mapped = map_sales_to_weeks(sales, weeks)
    n_unmapped = int(mapped["Week_End_Date"].isna().sum())
    print(f"  mapped rows={len(mapped) - n_unmapped:,}  unmapped={n_unmapped:,}")

    print("\n5) Aggregate weekly (Date = Week_End_Date kept for all options)...")
    weekly = aggregate_weekly(mapped)
    print(f"  weekly rows={len(weekly):,}  options={weekly['Option Code'].nunique()}  "
          f"sites={weekly['Site_Code'].nunique()}  weeks={weekly['Date'].nunique()}")

    print("\n6) Validate option-code totals...")
    cmp, summary = validate(sales, weekly, mapped)
    print(f"  all_ok={summary['all_ok']}  mismatches={summary['options_with_mismatch']}")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save
    weekly.to_parquet(OUT / "sales_weekly.parquet", index=False)
    # lighter csv sample / full if not huge
    if len(weekly) <= 2_000_000:
        weekly.to_csv(OUT / "sales_weekly.csv", index=False)
    else:
        weekly.head(100_000).to_csv(OUT / "sales_weekly_sample.csv", index=False)

    cmp.to_csv(OUT / "option_code_daily_vs_weekly_check.csv", index=False)
    mismatches = cmp.loc[~cmp["ok"]]
    mismatches.to_csv(OUT / "option_code_mismatches.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT / "validation_summary.csv", index=False)

    print(f"\nWrote {OUT / 'sales_weekly.parquet'}  "
          f"({(OUT / 'sales_weekly.parquet').stat().st_size/1e6:.1f} MB)")
    print(f"All outputs -> {OUT}")
    print("DONE — weekly sales ready.")
    return summary


if __name__ == "__main__":
    run()
