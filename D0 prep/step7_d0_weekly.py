"""
TCNS D0 - Step 7: Option-week availability + new launches -> AG x MRP group D0 weekly.

Input sales: step6 weekly sales (days over 60% discount already dropped in step2, then SKD 3-piece filter)
Input stock: step3 stock_weekly_8ag (weekly snapshots from 2025-04-28)

Option level (same as Tasva step7):
  Final_MRP          = MRP_corrected with highest Sales_Qty for the option
  stock_presence     = 1 if Stock_Qty >= 3 (site x option x week)
  site_share_pct     = site all-period Sales_Qty / option all-period Sales_Qty * 100
  availability_pct   = sum_sites(stock_presence * site_share_pct)   (0..100)
  variety_contrib    = availability_pct / 100
  first_sale_date    = first week with Sales_Qty > 0

Group = AG | Final_MRP, one row per Date x group:
  availability_pct   = sum(sales_qty_i * avail_i) / sum(sales_qty_i)
                       (simple mean when the group sold nothing that week)
  weighted_variety   = sum(variety_contrib_i)
  new_launch_options = options whose first_sale_date == Date

Weeks without a stock snapshot keep sales but availability / variety are NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys

_HERE = Path(__file__).resolve().parent
_CODE = _HERE if (_HERE / "paths.py").exists() else _HERE / "w_experiment" / "code"
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
from paths import AG_ORDER, out_dir, write_split

OUT = out_dir("step7_d0_weekly")
SALES_IN = out_dir("step6_filter_discount_gt60") / "sales_weekly_8ag_after_daily_filter.parquet"
STOCK_IN = out_dir("step3_filter_ag") / "stock_weekly_8ag.parquet"

STOCK_THRESHOLD = 3
TOL = 1e-4


def _norm_code(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def mrp_as_str(values: pd.Series) -> pd.Series:
    num = pd.to_numeric(values, errors="coerce")
    return num.map(
        lambda x: "" if pd.isna(x) else (str(int(x)) if float(x).is_integer() else str(x))
    ).astype("string")


def indian_fy_year(dates: pd.Series) -> pd.Series:
    start = np.where(dates.dt.month >= 4, dates.dt.year, dates.dt.year - 1)
    return pd.Series(
        [f"FY{str(s)[-2:]}-{str(s + 1)[-2:]}" for s in start], index=dates.index
    )


def load_sales() -> pd.DataFrame:
    cols = [
        "Date",
        "Site_Code",
        "Option Code",
        "AG",
        "scope_category",
        "MRP_corrected",
        "Sales_Qty",
        "Sales_NSV",
        "Sales_GSV",
    ]
    df = pd.read_parquet(SALES_IN, columns=cols)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df["Site_Code"] = _norm_code(df["Site_Code"])
    df["Option Code"] = _norm_code(df["Option Code"])
    for c in ["MRP_corrected", "Sales_Qty", "Sales_NSV", "Sales_GSV"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def final_mrp_map(sales: pd.DataFrame) -> pd.DataFrame:
    """One Final_MRP per option = MRP_corrected with most Sales_Qty (tie -> GSV)."""
    q = sales.groupby(["Option Code", "AG", "MRP_corrected"], as_index=False).agg(
        qty=("Sales_Qty", "sum"), gsv=("Sales_GSV", "sum")
    )
    q = q.sort_values(["Option Code", "qty", "gsv"], ascending=[True, False, False])
    m = q.drop_duplicates("Option Code")[["Option Code", "AG", "MRP_corrected"]]
    m = m.rename(columns={"MRP_corrected": "Final_MRP"})
    n = sales.groupby("Option Code")["MRP_corrected"].nunique().rename("n_mrps_on_option")
    m = m.merge(n, on="Option Code", how="left")
    m["group"] = m["AG"].astype(str) + " | " + mrp_as_str(m["Final_MRP"])
    return m


def first_sale_dates(sales: pd.DataFrame) -> pd.DataFrame:
    src = sales.loc[sales["Sales_Qty"] > 0, ["Date", "Option Code"]]
    return (
        src.groupby("Option Code", as_index=False)["Date"]
        .min()
        .rename(columns={"Date": "first_sale_date"})
    )


def load_stock(options: set[str]) -> pd.DataFrame:
    st = pd.read_parquet(STOCK_IN, columns=["Date", "Site_Code", "Option Code", "Stock Qty"])
    st["Option Code"] = _norm_code(st["Option Code"])
    st = st.loc[st["Option Code"].isin(options)].copy()
    st["Site_Code"] = _norm_code(st["Site_Code"])
    st["Date"] = pd.to_datetime(st["Date"]).dt.normalize()
    st["Stock_Qty"] = pd.to_numeric(st["Stock Qty"], errors="coerce").fillna(0.0)
    return st.groupby(["Date", "Site_Code", "Option Code"], as_index=False)["Stock_Qty"].sum()


def site_shares(sales: pd.DataFrame) -> pd.DataFrame:
    s = sales.groupby(["Option Code", "Site_Code"], as_index=False)["Sales_Qty"].sum()
    s = s.rename(columns={"Sales_Qty": "site_sales_qty"})
    tot = s.groupby("Option Code")["site_sales_qty"].transform("sum")
    s["option_sales_qty"] = tot
    s["site_share_pct"] = np.where(tot > 0, 100.0 * s["site_sales_qty"] / tot, 0.0)
    return s


def option_week_stock(stock: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    st = stock.merge(
        shares[["Option Code", "Site_Code", "site_share_pct"]],
        on=["Option Code", "Site_Code"],
        how="left",
    )
    st["site_share_pct"] = st["site_share_pct"].fillna(0.0)
    st["stock_presence"] = (st["Stock_Qty"] >= STOCK_THRESHOLD).astype("int8")
    st["availability_contrib"] = st["stock_presence"] * st["site_share_pct"]
    ow = st.groupby(["Date", "Option Code"], as_index=False).agg(
        availability_pct=("availability_contrib", "sum"),
        closing_stock=("Stock_Qty", "sum"),
        stock_stores=("Site_Code", "nunique"),
        available_stores=("stock_presence", "sum"),
    )
    ow["availability_pct"] = ow["availability_pct"].clip(0, 100)
    return ow


def build_option_week(
    sales: pd.DataFrame,
    ow_stock: pd.DataFrame,
    fmap: pd.DataFrame,
    first: pd.DataFrame,
    stock_dates: set,
) -> pd.DataFrame:
    ow_sales = sales.groupby(["Date", "Option Code"], as_index=False).agg(
        sales_qty=("Sales_Qty", "sum"),
        sales_nsv=("Sales_NSV", "sum"),
        sales_gsv=("Sales_GSV", "sum"),
        stores_with_sales=("Site_Code", "nunique"),
    )
    ow = ow_sales.merge(ow_stock, on=["Date", "Option Code"], how="outer")
    for c in ["sales_qty", "sales_nsv", "sales_gsv", "stores_with_sales"]:
        ow[c] = ow[c].fillna(0.0)

    ow["has_stock_snapshot"] = ow["Date"].isin(stock_dates)
    snap = ow["has_stock_snapshot"]
    for c in ["availability_pct", "closing_stock", "stock_stores", "available_stores"]:
        ow.loc[snap, c] = ow.loc[snap, c].fillna(0.0)
        ow.loc[~snap, c] = np.nan
    ow["variety_contrib"] = ow["availability_pct"] / 100.0
    ow["option_in_stock"] = np.where(snap, (ow["available_stores"] > 0).astype(float), np.nan)

    ow = ow.merge(fmap, on="Option Code", how="left").merge(first, on="Option Code", how="left")
    ow["is_new_launch"] = ow["Date"].eq(ow["first_sale_date"]).astype("int8")
    return ow.sort_values(["Date", "group", "Option Code"]).reset_index(drop=True)


def build_group_week(ow: pd.DataFrame) -> pd.DataFrame:
    t = ow.copy()
    t["w_x_avail"] = t["sales_qty"] * t["availability_pct"]
    t["has_sales"] = (t["sales_qty"] > 0).astype("int8")

    g = t.groupby(["group", "Date"], as_index=False).agg(
        AG=("AG", "first"),
        Final_MRP=("Final_MRP", "first"),
        has_stock_snapshot=("has_stock_snapshot", "first"),
        sales_qty=("sales_qty", "sum"),
        sales_nsv=("sales_nsv", "sum"),
        sales_gsv=("sales_gsv", "sum"),
        closing_stock=("closing_stock", "sum"),
        w_x_avail=("w_x_avail", "sum"),
        avail_mean=("availability_pct", "mean"),
        weighted_variety=("variety_contrib", "sum"),
        option_codes=("Option Code", "nunique"),
        options_with_sales=("has_sales", "sum"),
        options_in_stock=("option_in_stock", "sum"),
        new_launch_options=("is_new_launch", "sum"),
    )
    snap = g["has_stock_snapshot"].astype(bool)
    g["availability_pct"] = np.where(
        g["sales_qty"] > 0, g["w_x_avail"] / g["sales_qty"].where(g["sales_qty"] > 0), g["avail_mean"]
    )
    for c in ["availability_pct", "weighted_variety", "closing_stock", "options_in_stock"]:
        g.loc[~snap, c] = np.nan
    g = g.drop(columns=["w_x_avail", "avail_mean"])

    mrp = pd.to_numeric(g["Final_MRP"], errors="coerce")
    qty = g["sales_qty"]
    g["Original_Price"] = np.where(qty > 0, g["sales_gsv"] / qty.where(qty > 0), np.nan)
    g["Discount_Percentage"] = np.where(
        (mrp > 0) & (qty > 0), 100.0 * (mrp - g["Original_Price"]) / mrp, np.nan
    )

    first = (
        ow.loc[ow["is_new_launch"] == 1]
        .groupby("group")["Date"]
        .min()
        .rename("group_first_launch_date")
    )
    g = g.merge(first, on="group", how="left")

    d = g["Date"]
    g["Year"] = d.dt.year
    g["Month"] = d.dt.month
    g["Week"] = d.dt.isocalendar().week.astype(int)
    g["fy_year"] = indian_fy_year(d)
    g["Final_MRP"] = mrp_as_str(g["Final_MRP"])

    front = [
        "AG",
        "group",
        "Final_MRP",
        "fy_year",
        "Year",
        "Month",
        "Week",
        "Date",
        "has_stock_snapshot",
        "sales_qty",
        "sales_nsv",
        "sales_gsv",
        "Original_Price",
        "Discount_Percentage",
        "closing_stock",
        "availability_pct",
        "weighted_variety",
        "option_codes",
        "options_with_sales",
        "options_in_stock",
        "new_launch_options",
        "group_first_launch_date",
    ]
    g["_ag_ord"] = g["AG"].map({a: i for i, a in enumerate(AG_ORDER)})
    g["_mrp_num"] = pd.to_numeric(g["Final_MRP"], errors="coerce")
    g = g.sort_values(["_ag_ord", "_mrp_num", "Date"]).reset_index(drop=True)
    return g[front]


def group_summary(d0: pd.DataFrame, ow: pd.DataFrame) -> pd.DataFrame:
    opts = ow.loc[ow["sales_qty"] > 0].groupby("group")["Option Code"].nunique()
    s = d0.groupby(["AG", "group", "Final_MRP"], as_index=False).agg(
        weeks=("Date", "nunique"),
        weeks_with_sales=("sales_qty", lambda x: int((x > 0).sum())),
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        sales_qty=("sales_qty", "sum"),
        sales_gsv=("sales_gsv", "sum"),
        avg_availability_pct=("availability_pct", "mean"),
        avg_weighted_variety=("weighted_variety", "mean"),
        new_launch_options=("new_launch_options", "sum"),
    )
    s["unique_option_codes"] = s["group"].map(opts).fillna(0).astype(int)
    ag_gsv = s.groupby("AG")["sales_gsv"].transform("sum")
    s["gsv_share_within_AG_pct"] = (100.0 * s["sales_gsv"] / ag_gsv).round(4)
    s["_ag_ord"] = s["AG"].map({a: i for i, a in enumerate(AG_ORDER)})
    s = s.sort_values(["_ag_ord", "sales_gsv"], ascending=[True, False]).drop(columns="_ag_ord")
    return s.reset_index(drop=True)


def ag_summary(d0: pd.DataFrame) -> pd.DataFrame:
    snap = d0.loc[d0["has_stock_snapshot"]]
    a = d0.groupby("AG", as_index=False).agg(
        groups=("group", "nunique"),
        group_weeks=("group", "size"),
        sales_qty=("sales_qty", "sum"),
        sales_gsv=("sales_gsv", "sum"),
        new_launch_options=("new_launch_options", "sum"),
    )
    b = snap.groupby("AG", as_index=False).agg(
        group_weeks_with_stock=("group", "size"),
        avg_availability_pct=("availability_pct", "mean"),
        avg_weighted_variety=("weighted_variety", "mean"),
    )
    a = a.merge(b, on="AG", how="left")
    a["_ord"] = a["AG"].map({x: i for i, x in enumerate(AG_ORDER)})
    return a.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def validate(sales: pd.DataFrame, d0: pd.DataFrame) -> pd.DataFrame:
    s = sales.groupby("AG").agg(src_qty=("Sales_Qty", "sum"), src_gsv=("Sales_GSV", "sum"))
    d = d0.groupby("AG").agg(d0_qty=("sales_qty", "sum"), d0_gsv=("sales_gsv", "sum"))
    v = s.join(d, how="outer").fillna(0.0).reset_index()
    v.loc[len(v)] = ["TOTAL", *v[["src_qty", "src_gsv", "d0_qty", "d0_gsv"]].sum().tolist()]
    v["qty_ok"] = (v["src_qty"] - v["d0_qty"]).abs() <= TOL
    v["gsv_ok"] = (v["src_gsv"] - v["d0_gsv"]).abs() <= 1.0
    return v


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in [SALES_IN, STOCK_IN]:
        if not p.exists():
            raise FileNotFoundError(p)

    print("1) Load step5 sales...")
    sales = load_sales()
    options = set(sales["Option Code"].dropna().unique())
    print(f"  rows {len(sales):,} | options {len(options):,} | weeks {sales['Date'].nunique()}")

    print("2) Final_MRP per option + group = AG | Final_MRP ...")
    fmap = final_mrp_map(sales)
    print(
        f"  groups {fmap['group'].nunique():,} | "
        f"multi-MRP options {(fmap['n_mrps_on_option'] > 1).sum():,}"
    )

    print("3) First sale date per option...")
    first = first_sale_dates(sales)

    print("4) Load stock (step3, options in sales only)...")
    stock = load_stock(options)
    stock_dates = set(stock["Date"].unique())
    print(
        f"  site x option x week rows {len(stock):,} | snapshots {len(stock_dates)} "
        f"({min(stock_dates).date()} -> {max(stock_dates).date()})"
    )

    print("5) Site shares + option x week availability...")
    shares = site_shares(sales)
    ow_stock = option_week_stock(stock, shares)
    ow = build_option_week(sales, ow_stock, fmap, first, stock_dates)
    print(
        f"  option-week rows {len(ow):,} | "
        f"mean availability (snapshot weeks) {ow['availability_pct'].mean():.2f}%"
    )

    print("6) Group x week D0...")
    d0 = build_group_week(ow)
    n_dup = int(d0.duplicated(["group", "Date"]).sum())
    if n_dup:
        raise ValueError(f"D0 grain broken: {n_dup} duplicate group x Date rows")
    print(f"  D0 rows {len(d0):,} | groups {d0['group'].nunique()} | weeks {d0['Date'].nunique()}")

    val = validate(sales, d0)
    print("\n=== VALIDATION: D0 vs step5 sales ===")
    print(val.to_string(index=False))
    if not (val["qty_ok"].all() and val["gsv_ok"].all()):
        raise ValueError("D0 totals do not match step5 sales")

    gsum = group_summary(d0, ow)
    agsum = ag_summary(d0)
    print("\n=== AG SUMMARY ===")
    print(agsum.to_string(index=False))

    ow_out = ow.drop(columns=["n_mrps_on_option"]).copy()
    ow_out["Final_MRP"] = mrp_as_str(ow_out["Final_MRP"])
    ow_out.to_parquet(OUT / "option_week_availability.parquet", index=False)
    shares.to_csv(OUT / "site_option_share.csv", index=False)
    fmap.to_csv(OUT / "option_final_mrp_group_map.csv", index=False)
    d0.to_parquet(OUT / "d0_weekly_all_groups.parquet", index=False)
    print("\nSplitting D0 into the eight scope folders...")
    write_split(d0, "step7_d0_weekly", "d0_weekly.parquet")
    d0.to_csv(OUT / "d0_weekly_all_groups.csv", index=False)
    gsum.to_csv(OUT / "d0_group_summary.csv", index=False)
    agsum.to_csv(OUT / "d0_ag_summary.csv", index=False)
    val.to_csv(OUT / "d0_validation_vs_step5.csv", index=False)

    with pd.ExcelWriter(OUT / "d0_weekly_by_ag.xlsx", engine="openpyxl") as xw:
        agsum.to_excel(xw, sheet_name="AG_summary", index=False)
        gsum.to_excel(xw, sheet_name="group_summary", index=False)
        for ag in AG_ORDER:
            d0.loc[d0["AG"] == ag].to_excel(
                xw, sheet_name=ag.replace("|", "_")[:31], index=False
            )

    print(f"\nWrote outputs to: {OUT}")
    print("DONE - step7 D0 weekly complete.")
    return {"d0_rows": len(d0), "groups": int(d0["group"].nunique())}


if __name__ == "__main__":
    run()
