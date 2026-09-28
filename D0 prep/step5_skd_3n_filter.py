"""
TCNS D0 - Step 5: For SKD AGs only, keep Net Qty in {3N, 3 NUMBER, SET OF 3N}.

KURTA AGs are kept as-is (all Net Qty values).

Input:  step4b sales_weekly_8ag_mrp_corrected.parquet
Output: filtered sales + drop report vs step4b
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
from paths import AG_ORDER, out_dir, write_split

OUT = out_dir("step5_skd_3n_filter")
SALES_IN = out_dir("step4b_correct_mrp") / "sales_weekly_8ag_mrp_corrected.parquet"

# Allowed Net Qty for SKD categories (normalized)
SKD_NET_QTY_KEEP = {
    "3N",
    "3 NUMBER",
    "SET OF 3N",
}


def normalize_net_qty(s: pd.Series) -> pd.Series:
    """Uppercase, collapse whitespace: 'Set of 3N' / 'set of 3n' -> 'SET OF 3N'."""
    out = s.astype("string").str.strip().str.upper()
    out = out.str.replace(r"\s+", " ", regex=True)
    return out


def is_skd_ag(ag: pd.Series) -> pd.Series:
    return ag.astype("string").str.contains("SKD", case=False, na=False)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Output:", OUT)
    print("Input:", SALES_IN.exists(), SALES_IN)
    if not SALES_IN.exists():
        raise FileNotFoundError(f"Run step4b first. Missing: {SALES_IN}")

    print("\n1) Load step4b complete file...")
    df = pd.read_parquet(SALES_IN)
    before_rows = len(df)
    before_opts = int(df["Option Code"].nunique())
    before_qty = float(df["Sales_Qty"].sum())
    before_gsv = float(df["Sales_GSV"].sum())
    print(f"  rows {before_rows:,} | options {before_opts:,} | qty {before_qty:,.0f}")

    df = df.copy()
    df["Net_Qty_norm"] = normalize_net_qty(df["Net_Qty"])
    df["is_skd"] = is_skd_ag(df["AG"])

    # Keep: non-SKD all rows OR SKD with allowed Net Qty
    keep_mask = (~df["is_skd"]) | (df["Net_Qty_norm"].isin(SKD_NET_QTY_KEEP))
    dropped = df.loc[~keep_mask].copy()
    kept = df.loc[keep_mask].copy()

    after_rows = len(kept)
    after_opts = int(kept["Option Code"].nunique())
    after_qty = float(kept["Sales_Qty"].sum())
    after_gsv = float(kept["Sales_GSV"].sum())

    print("\n2) SKD Net Qty filter (3N / 3 NUMBER / SET OF 3N)...")
    print(f"  SKD rows before: {int(df['is_skd'].sum()):,}")
    print(f"  SKD rows kept:   {int(kept['is_skd'].sum()):,}")
    print(f"  SKD rows dropped:{int(dropped['is_skd'].sum()):,}")
    print(f"  KURTA rows (unchanged): {int((~df['is_skd']).sum()):,}")

    # Per-AG before/after
    rows = []
    for ag in AG_ORDER:
        b = df[df["AG"] == ag]
        a = kept[kept["AG"] == ag]
        d = dropped[dropped["AG"] == ag]
        rows.append(
            {
                "AG": ag,
                "scope_category": b["scope_category"].iloc[0] if len(b) else None,
                "is_skd": "SKD" in ag,
                "rows_before": len(b),
                "rows_after": len(a),
                "rows_dropped": len(d),
                "pct_rows_dropped": round(100 * len(d) / len(b), 2) if len(b) else 0.0,
                "options_before": int(b["Option Code"].nunique()) if len(b) else 0,
                "options_after": int(a["Option Code"].nunique()) if len(a) else 0,
                "options_dropped": int(
                    len(set(b["Option Code"]) - set(a["Option Code"]))
                )
                if len(b)
                else 0,
                "Sales_Qty_before": float(b["Sales_Qty"].sum()) if len(b) else 0.0,
                "Sales_Qty_after": float(a["Sales_Qty"].sum()) if len(a) else 0.0,
                "Sales_Qty_dropped": float(d["Sales_Qty"].sum()) if len(d) else 0.0,
                "Sales_GSV_before": float(b["Sales_GSV"].sum()) if len(b) else 0.0,
                "Sales_GSV_after": float(a["Sales_GSV"].sum()) if len(a) else 0.0,
                "Sales_GSV_dropped": float(d["Sales_GSV"].sum()) if len(d) else 0.0,
            }
        )
    by_ag = pd.DataFrame(rows)

    # What Net Qty values were dropped from SKD
    drop_nq = (
        dropped.groupby(["AG", "Net_Qty_norm"], as_index=False)
        .agg(
            rows=("Option Code", "size"),
            Sales_Qty=("Sales_Qty", "sum"),
            options=("Option Code", "nunique"),
        )
        .sort_values(["AG", "Sales_Qty"], ascending=[True, False])
    )

    print("\n=== DROP BY AG (vs step4b) ===")
    print(
        by_ag[
            [
                "AG",
                "is_skd",
                "rows_before",
                "rows_after",
                "rows_dropped",
                "pct_rows_dropped",
                "options_before",
                "options_after",
                "options_dropped",
            ]
        ].to_string(index=False)
    )

    overall = {
        "rows_before_step4b": before_rows,
        "rows_after_step5": after_rows,
        "rows_dropped": before_rows - after_rows,
        "pct_rows_dropped": round(100 * (before_rows - after_rows) / before_rows, 2),
        "options_before": before_opts,
        "options_after": after_opts,
        "options_dropped": before_opts - after_opts,
        "Sales_Qty_before": before_qty,
        "Sales_Qty_after": after_qty,
        "Sales_Qty_dropped": before_qty - after_qty,
        "pct_Sales_Qty_dropped": round(100 * (before_qty - after_qty) / before_qty, 2)
        if before_qty
        else 0.0,
        "Sales_GSV_before": before_gsv,
        "Sales_GSV_after": after_gsv,
        "Sales_GSV_dropped": before_gsv - after_gsv,
        "skd_net_qty_keep": " | ".join(sorted(SKD_NET_QTY_KEEP)),
        "filter_applies_to": "AG containing SKD only; KURTA unchanged",
    }
    print("\n=== OVERALL vs step4b ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    # Drop helper cols from final file (keep Net_Qty_norm useful? keep it for audit)
    # Keep Net_Qty_norm; drop is_skd
    kept_out = kept.drop(columns=["is_skd"])

    print("\n3) Save filtered file...")
    kept_out.to_parquet(OUT / "sales_weekly_8ag_skd_3n_filtered.parquet", index=False)
    print("\nSplitting step 5 sales into the four scope folders...")
    write_split(kept_out, "step5_skd_3n_filter", "sales_weekly_skd_3n.parquet")
    kept_out.to_csv(OUT / "sales_weekly_8ag_skd_3n_filtered.csv", index=False)
    dropped.drop(columns=["is_skd"]).to_csv(OUT / "sales_rows_dropped_skd_net_qty.csv", index=False)
    by_ag.to_csv(OUT / "step5_drop_by_ag.csv", index=False)
    drop_nq.to_csv(OUT / "step5_dropped_net_qty_breakdown.csv", index=False)
    pd.DataFrame([overall]).to_csv(OUT / "step5_summary.csv", index=False)

    with pd.ExcelWriter(OUT / "step5_skd_3n_filter_report.xlsx", engine="openpyxl") as xw:
        pd.DataFrame([overall]).to_excel(xw, sheet_name="Summary", index=False)
        by_ag.to_excel(xw, sheet_name="Drop_by_AG", index=False)
        drop_nq.to_excel(xw, sheet_name="Dropped_NetQty", index=False)

    print(f"\nWrote {OUT / 'sales_weekly_8ag_skd_3n_filtered.parquet'}")
    print(f"Wrote {OUT / 'sales_weekly_8ag_skd_3n_filtered.csv'}")
    print("DONE - Step 5 SKD 3N filter complete.")
    return overall


if __name__ == "__main__":
    run()
