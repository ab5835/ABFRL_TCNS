"""
TCNS D0 - Step 4b: Add NEW column MRP_corrected (do not overwrite MRP).

Rule (100-buck Indian retail ladder ending in 99):
  MRP_corrected = round((MRP - 99) / 100) * 100 + 99

Example: 1573 -> 1599

Outputs:
  - complete sales with original MRP + new MRP_corrected
  - AG x MRP_corrected: unique option codes + sales
  - AG x MRP_corrected x Option Code detail
  - AG rollup: unique new MRPs + unique option codes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
from paths import AG_ORDER, out_dir, write_split

STEP4_OUT = out_dir("step4_merge_net_qty")
OUT = out_dir("step4b_correct_mrp")

# Always start from step4 file that still has ORIGINAL MRP (not prior 4b overwrite)
SALES_IN_CANDIDATES = [
    STEP4_OUT / "sales_weekly_8ag_with_net_qty.parquet",
    out_dir("step3_filter_ag") / "sales_weekly_8ag.parquet",
]


def correct_mrp_to_x99(mrp: pd.Series) -> pd.Series:
    """Snap to nearest ...99 with 100 gap. Floor at 99."""
    x = pd.to_numeric(mrp, errors="coerce")
    steps = np.round((x - 99.0) / 100.0)
    out = steps * 100.0 + 99.0
    out = out.where(x.notna(), np.nan)
    return out.clip(lower=99.0)


def load_sales() -> tuple[pd.DataFrame, Path]:
    for p in SALES_IN_CANDIDATES:
        if p.exists():
            return pd.read_parquet(p), p
    raise FileNotFoundError(
        "No step4/step3 sales parquet found:\n" + "\n".join(map(str, SALES_IN_CANDIDATES))
    )


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    sales, src = load_sales()
    print("Output:", OUT)
    print("Input:", src)
    print(f"  rows {len(sales):,} | options {sales['Option Code'].nunique():,}")

    sales = sales.copy()

    # Drop leftover cols from a prior overwrite run, if any
    for c in ["MRP_raw", "MRP_was_corrected", "MRP_corrected"]:
        if c in sales.columns:
            sales = sales.drop(columns=[c])

    # ORIGINAL MRP stays as MRP
    sales["MRP"] = pd.to_numeric(sales["MRP"], errors="coerce")

    # NEW column only
    sales["MRP_corrected"] = correct_mrp_to_x99(sales["MRP"])
    sales["MRP_was_corrected"] = (sales["MRP_corrected"] - sales["MRP"]).abs() > 0.01

    n_changed = int(sales["MRP_was_corrected"].sum())
    print(
        f"\n1) Added NEW column MRP_corrected (original MRP unchanged): "
        f"{n_changed:,} / {len(sales):,} rows differ "
        f"({100 * n_changed / len(sales):.1f}%)"
    )
    print(
        f"  unique MRP (original): {sales['MRP'].nunique()} | "
        f"unique MRP_corrected: {sales['MRP_corrected'].nunique()}"
    )
    print(f"  columns: MRP={('MRP' in sales.columns)} MRP_corrected={('MRP_corrected' in sales.columns)}")

    # mapping audit
    map_df = (
        sales.groupby(["MRP", "MRP_corrected"], as_index=False)
        .agg(
            rows=("Option Code", "size"),
            Sales_Qty=("Sales_Qty", "sum"),
            options=("Option Code", "nunique"),
        )
        .sort_values("Sales_Qty", ascending=False)
    )
    map_df["delta"] = map_df["MRP_corrected"] - map_df["MRP"]
    map_df.to_csv(OUT / "mrp_to_mrp_corrected_map.csv", index=False)

    print("\n2) Save complete file (original MRP + new MRP_corrected)...")
    out_pq = OUT / "sales_weekly_8ag_mrp_corrected.parquet"
    sales.to_parquet(out_pq, index=False)
    print("\nSplitting step 4b sales into the four scope folders...")
    write_split(sales, "step4b_correct_mrp", "sales_weekly_mrp_corrected.parquet")
    STEP4_OUT.mkdir(parents=True, exist_ok=True)
    sales.to_parquet(STEP4_OUT / "sales_weekly_8ag_mrp_corrected.parquet", index=False)
    sales.head(100_000).to_csv(OUT / "sales_weekly_8ag_mrp_corrected_sample.csv", index=False)

    print("\n3) AG x MRP_corrected summary (unique option codes + sales)...")
    ag_mrp = (
        sales.groupby(["AG", "scope_category", "MRP_corrected"], dropna=False, as_index=False)
        .agg(
            total_Sales_Qty=("Sales_Qty", "sum"),
            total_Sales_GSV=("Sales_GSV", "sum"),
            unique_option_codes=("Option Code", "nunique"),
            unique_sites=("Site_Code", "nunique"),
            weekly_rows=("Option Code", "size"),
        )
    )
    ag_mrp["AG"] = pd.Categorical(ag_mrp["AG"], categories=AG_ORDER, ordered=True)
    ag_mrp = ag_mrp.sort_values(["AG", "MRP_corrected"]).reset_index(drop=True)
    ag_mrp["AG"] = ag_mrp["AG"].astype(str)
    ag_mrp.to_csv(OUT / "ag_mrp_corrected_option_counts.csv", index=False)

    print("\n4) AG x MRP_corrected x Option Code detail...")
    ag_mrp_opt = (
        sales.groupby(["AG", "scope_category", "MRP_corrected", "Option Code"], as_index=False)
        .agg(
            total_Sales_Qty=("Sales_Qty", "sum"),
            total_Sales_GSV=("Sales_GSV", "sum"),
            weekly_rows=("Option Code", "size"),
        )
        .sort_values(["AG", "MRP_corrected", "Option Code"])
    )
    ag_mrp_opt.to_csv(OUT / "ag_mrp_corrected_option_codes_detail.csv", index=False)
    # also parquet for size
    ag_mrp_opt.to_parquet(OUT / "ag_mrp_corrected_option_codes_detail.parquet", index=False)
    print(f"  AG x MRP_corrected points: {len(ag_mrp):,}")
    print(f"  AG x MRP_corrected x Option rows: {len(ag_mrp_opt):,}")

    print("\n5) AG rollup: unique new MRPs + option codes...")
    rows = []
    for ag in AG_ORDER:
        g = sales[sales["AG"] == ag]
        rows.append(
            {
                "AG": ag,
                "scope_category": g["scope_category"].iloc[0] if len(g) else None,
                "unique_MRP_original": int(g["MRP"].nunique()) if len(g) else 0,
                "unique_MRP_corrected": int(g["MRP_corrected"].nunique()) if len(g) else 0,
                "unique_option_codes": int(g["Option Code"].nunique()) if len(g) else 0,
                "total_Sales_Qty": float(g["Sales_Qty"].sum()) if len(g) else 0.0,
                "total_Sales_GSV": float(g["Sales_GSV"].sum()) if len(g) else 0.0,
            }
        )
    ag_tot = pd.DataFrame(rows)
    ag_tot.to_csv(OUT / "ag_unique_mrp_corrected.csv", index=False)

    with pd.ExcelWriter(OUT / "ag_mrp_corrected_summary.xlsx", engine="openpyxl") as xw:
        ag_tot.to_excel(xw, sheet_name="AG_unique_MRP_options", index=False)
        ag_mrp.to_excel(xw, sheet_name="AG_x_MRP_option_counts", index=False)
        # detail may be large - cap sheet rows; full detail is in CSV/parquet
        for ag in AG_ORDER:
            sub = ag_mrp[ag_mrp["AG"] == ag]
            sheet = ("cnt_" + ag.replace("|", "_"))[:31]
            sub.to_excel(xw, sheet_name=sheet, index=False)
        # sample of option detail (top by qty per AG)
        sample_parts = []
        for ag in AG_ORDER:
            sub = ag_mrp_opt[ag_mrp_opt["AG"] == ag].nlargest(500, "total_Sales_Qty")
            sample_parts.append(sub)
        pd.concat(sample_parts, ignore_index=True).to_excel(
            xw, sheet_name="option_detail_top500_eachAG", index=False
        )

    print("\n=== AG TABLE: unique MRP_corrected + option codes ===")
    print(ag_tot.to_string(index=False))

    print("\n=== SAMPLE: AG x MRP_corrected with unique option codes ===")
    print(ag_mrp.head(20).to_string(index=False))

    overall = {
        "mrp_original_column_kept": True,
        "mrp_corrected_new_column": True,
        "rows": len(sales),
        "rows_where_corrected_differs": n_changed,
        "unique_mrp_original": int(sales["MRP"].nunique()),
        "unique_mrp_corrected": int(sales["MRP_corrected"].nunique()),
        "ag_mrp_corrected_points": len(ag_mrp),
        "ag_mrp_option_detail_rows": len(ag_mrp_opt),
        "rule": "round((MRP-99)/100)*100+99",
    }
    pd.DataFrame([overall]).to_csv(OUT / "step4b_summary.csv", index=False)
    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {out_pq}")
    print("DONE.")
    return overall


if __name__ == "__main__":
    run()
