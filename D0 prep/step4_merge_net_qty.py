"""
TCNS D0 - Step 4: Merge Item Master Net Qty (+ attrs) onto Step-3 8-AG weekly sales.

Then summarize each AG x MRP:
  - total Sales_Qty / Sales_GSV
  - unique Option Codes at that MRP point

Outputs under step4_merge_net_qty/output/
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pandas as pd

import sys

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
from paths import (
    AG_ORDER,
    ITEM_MASTER,
    ITEM_MASTER_FALLBACK_CSV,
    out_dir,
    write_split,
)

OUT = out_dir("step4_merge_net_qty")
SALES_8AG = out_dir("step3_filter_ag") / "sales_weekly_8ag.parquet"

IM_ATTR_COLS = [
    "AG",
    "AX Aricle",
    "Cross-plant CM",
    "Dv",
    "Final Cat",
    "Occasion",
    "Net Qty",
    "Colour",
    "Season",
    "Season Bracket",
    "SEASON CAT",
    "Product name",
    "Usability",
    "Sub Cat",
    "Brand",
    "Amount",
]


def _clean_str(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.mask(out.isin(["<NA>", "NAN", "NONE", ""]))


def _mode_or_first(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return pd.NA
    return s.value_counts().index[0]


def load_item_master(path: Path = ITEM_MASTER) -> pd.DataFrame:
    """Read Item Master; temp copy or prior CSV if locked."""
    try:
        return pd.read_excel(path, engine="pyxlsb", usecols=lambda c: c in IM_ATTR_COLS)
    except (PermissionError, OSError) as e:
        print(f"  Item Master direct read failed ({type(e).__name__})")
        tmp = Path(tempfile.gettempdir()) / "TCNS_Item_Master_step4.xlsb"
        copied = False
        for copy_fn in (shutil.copy2, shutil.copyfile):
            try:
                copy_fn(path, tmp)
                copied = True
                break
            except (PermissionError, OSError):
                continue
        if copied:
            print(f"  using temp copy: {tmp}")
            return pd.read_excel(tmp, engine="pyxlsb", usecols=lambda c: c in IM_ATTR_COLS)
        if ITEM_MASTER_FALLBACK_CSV.exists():
            print(f"  using fallback CSV (Net Qty may be missing): {ITEM_MASTER_FALLBACK_CSV}")
            fb = pd.read_csv(ITEM_MASTER_FALLBACK_CSV)
            return pd.DataFrame(
                {
                    "AG": fb.get("AG"),
                    "AX Aricle": fb.get("style_key"),
                    "Cross-plant CM": fb.get("style_key"),
                    "Dv": fb.get("Dv"),
                    "Final Cat": fb.get("Final_Cat"),
                    "Occasion": fb.get("Occasion"),
                    "Net Qty": pd.NA,
                    "Colour": pd.NA,
                    "Season": fb.get("Season"),
                    "Season Bracket": pd.NA,
                    "SEASON CAT": pd.NA,
                    "Product name": pd.NA,
                    "Usability": pd.NA,
                    "Sub Cat": pd.NA,
                    "Brand": fb.get("brand"),
                    "Amount": pd.NA,
                }
            )
        raise PermissionError(f"Cannot read Item Master: {path}") from e


def build_option_attrs(im: pd.DataFrame) -> pd.DataFrame:
    """One row per Option Code with Net Qty + key Item Master attributes."""
    im = im.copy()
    im["AG_n"] = _clean_str(im["AG"]).str.upper()
    im = im[im["AG_n"].isin(AG_ORDER)].copy()
    im["Net_Qty"] = _clean_str(im["Net Qty"])

    parts = []
    for col in ["Cross-plant CM", "AX Aricle"]:
        if col not in im.columns:
            continue
        keep = [
            "AG_n",
            "Net_Qty",
            "Dv",
            "Final Cat",
            "Occasion",
            "Colour",
            "Season",
            "Season Bracket",
            "SEASON CAT",
            "Product name",
            "Usability",
            "Sub Cat",
            "Brand",
            "Amount",
            col,
        ]
        keep = [c for c in keep if c in im.columns]
        t = im[keep].copy()
        t["Option Code"] = _clean_str(t[col]).str.upper()
        t = t[t["Option Code"].notna()]
        parts.append(t.drop(columns=[col]))

    keys = pd.concat(parts, ignore_index=True)

    agg = {
        "AG_n": _mode_or_first,
        "Net_Qty": _mode_or_first,
        "Dv": _mode_or_first,
        "Final Cat": _mode_or_first,
        "Occasion": _mode_or_first,
        "Colour": _mode_or_first,
        "Season": _mode_or_first,
        "Season Bracket": _mode_or_first,
        "SEASON CAT": _mode_or_first,
        "Product name": _mode_or_first,
        "Usability": _mode_or_first,
        "Sub Cat": _mode_or_first,
        "Brand": _mode_or_first,
        "Amount": _mode_or_first,
    }
    agg = {k: v for k, v in agg.items() if k in keys.columns}

    opt = (
        keys.groupby("Option Code", as_index=False)
        .agg({k: _mode_or_first for k in agg})
    )
    opt = opt.rename(
        columns={
            "AG_n": "AG_im",
            "Dv": "Brand_Dv_im",
            "Final Cat": "Final_Cat_im",
            "Occasion": "Occasion_im",
            "Colour": "Colour_im",
            "Season": "Season_im",
            "Season Bracket": "Season_Bracket_im",
            "SEASON CAT": "Season_Cat_im",
            "Product name": "Product_name_im",
            "Usability": "Usability_im",
            "Sub Cat": "Sub_Cat_im",
            "Brand": "Brand_im",
            "Amount": "Amount_im",
        }
    )
    opt["has_net_qty"] = opt["Net_Qty"].notna()
    return opt


def summarize_ag_mrp(df: pd.DataFrame) -> pd.DataFrame:
    """One row per AG x MRP: total sales + unique option codes."""
    g = (
        df.groupby(["AG", "scope_category", "MRP"], dropna=False, as_index=False)
        .agg(
            total_Sales_Qty=("Sales_Qty", "sum"),
            total_Sales_GSV=("Sales_GSV", "sum"),
            unique_option_codes=("Option Code", "nunique"),
            unique_sites=("Site_Code", "nunique"),
            weekly_rows=("Option Code", "size"),
        )
    )
    # stable AG order
    g["AG"] = pd.Categorical(g["AG"], categories=AG_ORDER, ordered=True)
    g = g.sort_values(["AG", "MRP"]).reset_index(drop=True)
    g["AG"] = g["AG"].astype(str)
    return g


def summarize_ag_totals(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ag in AG_ORDER:
        g = df[df["AG"] == ag]
        rows.append(
            {
                "AG": ag,
                "scope_category": g["scope_category"].iloc[0] if len(g) else None,
                "unique_MRPs": int(g["MRP"].nunique()) if len(g) else 0,
                "unique_option_codes": int(g["Option Code"].nunique()) if len(g) else 0,
                "total_Sales_Qty": float(g["Sales_Qty"].sum()) if len(g) else 0.0,
                "total_Sales_GSV": float(g["Sales_GSV"].sum()) if len(g) else 0.0,
                "pct_rows_with_net_qty": round(100 * g["has_net_qty"].mean(), 2) if len(g) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Output:", OUT)
    print("Sales 8AG:", SALES_8AG.exists(), SALES_8AG)
    print("Item Master:", ITEM_MASTER.exists(), ITEM_MASTER)

    if not SALES_8AG.exists():
        raise FileNotFoundError(f"Run step3 first. Missing: {SALES_8AG}")

    print("\n1) Load Item Master attributes (Net Qty)...")
    im = load_item_master()
    opt_attrs = build_option_attrs(im)
    print(f"  option attrs: {len(opt_attrs):,} | with Net_Qty: {int(opt_attrs['has_net_qty'].sum()):,}")
    print("  Net_Qty top values:")
    print(opt_attrs["Net_Qty"].value_counts(dropna=False).head(12).to_string())
    opt_attrs.to_csv(OUT / "item_master_option_net_qty_attrs.csv", index=False)

    print("\n2) Merge Net Qty onto Step-3 weekly sales (LEFT on Option Code)...")
    sales = pd.read_parquet(SALES_8AG)
    sales["Option Code"] = sales["Option Code"].astype("string").str.strip().str.upper()
    before_cols = set(sales.columns)

    # drop prior IM attrs if re-run
    drop_prior = [
        c
        for c in sales.columns
        if c
        in {
            "Net_Qty",
            "has_net_qty",
            "AG_im",
            "Brand_Dv_im",
            "Final_Cat_im",
            "Occasion_im",
            "Colour_im",
            "Season_im",
            "Season_Bracket_im",
            "Season_Cat_im",
            "Product_name_im",
            "Usability_im",
            "Sub_Cat_im",
            "Brand_im",
            "Amount_im",
        }
    ]
    if drop_prior:
        sales = sales.drop(columns=drop_prior)

    merged = sales.merge(opt_attrs, on="Option Code", how="left")
    # prefer step3 AG if present; keep AG_im for audit
    if "AG" not in merged.columns and "AG_im" in merged.columns:
        merged["AG"] = merged["AG_im"]
    merged["has_net_qty"] = merged["Net_Qty"].notna()
    merged["MRP"] = pd.to_numeric(merged["MRP"], errors="coerce")

    print(
        f"  rows {len(merged):,} | options {merged['Option Code'].nunique():,} | "
        f"with Net_Qty {int(merged['has_net_qty'].sum()):,} "
        f"({100 * merged['has_net_qty'].mean():.1f}% rows)"
    )
    new_cols = sorted(set(merged.columns) - before_cols)
    print(f"  added cols: {new_cols}")

    print("\n3) Save enriched sales...")
    merged.to_parquet(OUT / "sales_weekly_8ag_with_net_qty.parquet", index=False)
    print("\nSplitting step 4 sales into the four scope folders...")
    write_split(merged, "step4_merge_net_qty", "sales_weekly_with_net_qty.parquet")
    # full CSV may be large; write sample + keep parquet canonical
    merged.head(100_000).to_csv(OUT / "sales_weekly_8ag_with_net_qty_sample.csv", index=False)

    print("\n4) AG x MRP summary...")
    ag_mrp = summarize_ag_mrp(merged)
    ag_tot = summarize_ag_totals(merged)
    ag_mrp.to_csv(OUT / "ag_mrp_sales_summary.csv", index=False)
    ag_tot.to_csv(OUT / "ag_totals_summary.csv", index=False)

    # wide-ish helper: AG as rows block, MRP column explicit
    # also Excel with one sheet per AG for readability
    with pd.ExcelWriter(OUT / "ag_mrp_sales_summary.xlsx", engine="openpyxl") as xw:
        ag_mrp.to_excel(xw, sheet_name="AG_x_MRP_all", index=False)
        ag_tot.to_excel(xw, sheet_name="AG_totals", index=False)
        for ag in AG_ORDER:
            sub = ag_mrp[ag_mrp["AG"] == ag].copy()
            # sheet name max 31 chars
            sheet = ag.replace("|", "_")[:31]
            sub.to_excel(xw, sheet_name=sheet, index=False)

    print("\n=== AG TOTALS ===")
    print(ag_tot.to_string(index=False))
    print(f"\n=== AG x MRP rows: {len(ag_mrp):,} ===")
    print(ag_mrp.head(20).to_string(index=False))

    overall = {
        "rows": len(merged),
        "options": int(merged["Option Code"].nunique()),
        "rows_with_net_qty": int(merged["has_net_qty"].sum()),
        "pct_rows_with_net_qty": round(100 * merged["has_net_qty"].mean(), 2),
        "unique_ag": int(merged["AG"].nunique()),
        "unique_mrp": int(merged["MRP"].nunique()),
        "ag_mrp_points": len(ag_mrp),
    }
    pd.DataFrame([overall]).to_csv(OUT / "step4_summary.csv", index=False)
    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {OUT / 'sales_weekly_8ag_with_net_qty.parquet'}")
    print(f"Wrote {OUT / 'ag_mrp_sales_summary.xlsx'}")
    print("DONE - Step 4 Net Qty merge + AG x MRP summary complete.")
    return overall


if __name__ == "__main__":
    run()
