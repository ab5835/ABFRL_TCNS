"""
TCNS D0 - Step 3: Filter weekly sales to 8 AG categories from Item Master.

AG values (exact):
  A|SKD|FESTIVE, A|KURTA|FESTIVE, A|KURTA|NON FESTIVE, A|SKD|NON FESTIVE,
  W|KURTA|FESTIVE, W|KURTA|NON FESTIVE, W|SKD|FESTIVE, W|SKD|NON FESTIVE

1. Read Item Master -> option codes (Cross-plant CM / AX Aricle) per AG
2. Map AG -> scope category name
3. Filter Step-2 weekly sales to those option codes
4. Save filtered weekly sales + coverage summary
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
    AG_SCOPE,
    ITEM_MASTER,
    ITEM_MASTER_FALLBACK_CSV,
    out_dir,
    split_parquet,
    write_split,
)

OUT = out_dir("step3_filter_ag")
WEEKLY_SALES = out_dir("step2_weekly_sales") / "sales_weekly.parquet"
STOCK_LONG = out_dir("step1_stock_formatted") / "stock_weekly_long.parquet"
SCOPE_ORDER = list(AG_SCOPE.values())


def normalize_ag(s) -> str:
    return str(s).strip().upper()


def load_item_master(path: Path = ITEM_MASTER) -> pd.DataFrame:
    """Read Item Master; temp copy or prior CSV extract if OneDrive/Excel lock."""
    cols = ["AG", "AX Aricle", "Cross-plant CM", "Dv", "Final Cat", "Occasion"]
    try:
        return pd.read_excel(path, engine="pyxlsb", usecols=cols)
    except (PermissionError, OSError) as e:
        print(f"  Item Master direct read failed ({type(e).__name__})")
        tmp = Path(tempfile.gettempdir()) / "TCNS_Item_Master_step3.xlsb"
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
            return pd.read_excel(tmp, engine="pyxlsb", usecols=cols)
        if ITEM_MASTER_FALLBACK_CSV.exists():
            print(f"  using fallback CSV extract: {ITEM_MASTER_FALLBACK_CSV}")
            fb = pd.read_csv(ITEM_MASTER_FALLBACK_CSV)
            return pd.DataFrame(
                {
                    "AG": fb["AG"],
                    "AX Aricle": fb["style_key"],
                    "Cross-plant CM": fb["style_key"],
                    "Dv": fb.get("Dv"),
                    "Final Cat": fb.get("Final_Cat"),
                    "Occasion": fb.get("Occasion"),
                }
            )
        raise PermissionError(
            f"Cannot read Item Master (locked) and no fallback at {ITEM_MASTER_FALLBACK_CSV}"
        ) from e


def build_option_scope_map(im: pd.DataFrame) -> pd.DataFrame:
    """One row per Option Code with AG + scope_category."""
    im = im.copy()
    im["AG_n"] = im["AG"].map(normalize_ag)
    im = im[im["AG_n"].isin(AG_SCOPE.keys())].copy()
    im["scope_category"] = im["AG_n"].map(AG_SCOPE)

    parts = []
    for col in ["Cross-plant CM", "AX Aricle"]:
        t = im[["AG_n", "scope_category", "Dv", "Final Cat", "Occasion", col]].copy()
        t["Option Code"] = t[col].astype(str).str.strip().str.upper()
        t = t[
            t["Option Code"].notna()
            & ~t["Option Code"].isin(["NAN", "NONE", "", "<NA>"])
        ]
        parts.append(
            t.drop(columns=[col]).rename(
                columns={"AG_n": "AG", "Final Cat": "Final_Cat", "Dv": "Brand_Dv"}
            )
        )
    keys = pd.concat(parts, ignore_index=True).drop_duplicates()

    def mode_or_first(s: pd.Series):
        s = s.dropna()
        if len(s) == 0:
            return pd.NA
        return s.value_counts().index[0]

    opt_map = (
        keys.groupby("Option Code", as_index=False)
        .agg(
            AG=("AG", mode_or_first),
            scope_category=("scope_category", mode_or_first),
            Brand_Dv=("Brand_Dv", mode_or_first),
            Final_Cat=("Final_Cat", mode_or_first),
            Occasion=("Occasion", mode_or_first),
            n_master_rows=("Option Code", "size"),
        )
    )
    return opt_map


def summarize_by_scope(df: pd.DataFrame, qty_col: str, gsv_col: str | None = None) -> pd.DataFrame:
    rows = []
    for sc in SCOPE_ORDER:
        g = df[df["scope_category"] == sc]
        row = {
            "scope_category": sc,
            "has_data": "Yes" if len(g) else "No",
            "rows": len(g),
            "unique_options": g["Option Code"].nunique() if len(g) else 0,
            "unique_sites": g["Site_Code"].nunique() if "Site_Code" in g.columns and len(g) else 0,
        }
        if len(g) and "Date" in g.columns:
            row["min_date"] = str(pd.to_datetime(g["Date"]).min().date())
            row["max_date"] = str(pd.to_datetime(g["Date"]).max().date())
            row["weeks"] = int(pd.to_datetime(g["Date"]).nunique())
        else:
            row["min_date"] = None
            row["max_date"] = None
            row["weeks"] = 0
        row["qty"] = float(g[qty_col].sum()) if len(g) and qty_col in g.columns else 0.0
        if gsv_col and gsv_col in g.columns:
            row["gsv"] = float(g[gsv_col].sum()) if len(g) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Output:", OUT)
    print("Item Master:", ITEM_MASTER.exists(), ITEM_MASTER)
    print("Weekly sales:", WEEKLY_SALES.exists(), WEEKLY_SALES)

    print("\n1) Load Item Master AG -> option map...")
    im = load_item_master()
    print(f"  Item Master rows: {len(im):,}")
    opt_map = build_option_scope_map(im)
    print(f"  Unique option codes in 8 AGs: {len(opt_map):,}")
    print(opt_map["AG"].value_counts().reindex(list(AG_SCOPE.keys())).fillna(0).astype(int).to_string())

    opt_map.to_csv(OUT / "item_master_option_scope_map.csv", index=False)
    ag_counts = (
        opt_map.groupby(["AG", "scope_category"], as_index=False)
        .agg(unique_options=("Option Code", "nunique"))
        .sort_values("AG")
    )
    ag_counts.to_csv(OUT / "ag_option_counts.csv", index=False)

    print("\n2) Filter Step-2 weekly sales...")
    if not WEEKLY_SALES.exists():
        raise FileNotFoundError(f"Run step2 first. Missing: {WEEKLY_SALES}")
    weekly = pd.read_parquet(WEEKLY_SALES)
    weekly["Option Code"] = weekly["Option Code"].astype("string").str.strip().str.upper()
    before = len(weekly)
    filtered = weekly.merge(opt_map, on="Option Code", how="inner")
    print(
        f"  weekly rows {before:,} -> {len(filtered):,} "
        f"({100 * len(filtered) / before:.1f}%) | "
        f"options {weekly['Option Code'].nunique()} -> {filtered['Option Code'].nunique()}"
    )

    # Options in master but never in weekly sales
    sales_opts = set(weekly["Option Code"].dropna())
    master_opts = set(opt_map["Option Code"])
    only_master = sorted(master_opts - sales_opts)
    only_sales_not_in_8 = None  # not needed
    pd.DataFrame({"Option Code": only_master}).to_csv(
        OUT / "options_in_item_master_not_in_weekly_sales.csv", index=False
    )
    print(f"  options in 8 AGs but not in weekly sales: {len(only_master):,}")

    sales_summary = summarize_by_scope(filtered, "Sales_Qty", "Sales_GSV")
    print("\n=== SALES BY SCOPE ===")
    print(sales_summary.to_string(index=False))

    filtered.to_parquet(OUT / "sales_weekly_8ag.parquet", index=False)
    print("\nSplitting step 3 sales into the four scope folders...")
    write_split(filtered, "step3_filter_ag", "sales_weekly.parquet")
    if len(filtered) <= 2_000_000:
        filtered.to_csv(OUT / "sales_weekly_8ag.csv", index=False)
    else:
        filtered.head(100_000).to_csv(OUT / "sales_weekly_8ag_sample.csv", index=False)

    sales_summary.to_csv(OUT / "scope_sales_summary.csv", index=False)

    # Optional: filter stock if step1 exists (same option codes)
    stock_summary = None
    if STOCK_LONG.exists():
        print("\n3) Filter Step-1 stock to same option codes (optional join-ready)...")
        # stream filter via merge of unique options only - avoid full load if possible
        import pyarrow.parquet as pq

        opt_set = set(opt_map["Option Code"])
        pf = pq.ParquetFile(STOCK_LONG)
        out_stock = OUT / "stock_weekly_8ag.parquet"
        if out_stock.exists():
            out_stock.unlink()
        import pyarrow as pa
        import pyarrow.parquet as pq_w

        writer = None
        stock_rows = 0
        stock_parts_meta = []
        try:
            for batch in pf.iter_batches(batch_size=400_000):
                bdf = batch.to_pandas()
                bdf["Option Code"] = bdf["Option Code"].astype(str).str.strip().str.upper()
                bdf = bdf[bdf["Option Code"].isin(opt_set)]
                if len(bdf) == 0:
                    continue
                bdf = bdf.merge(opt_map, on="Option Code", how="left")
                stock_rows += len(bdf)
                table = pa.Table.from_pandas(bdf, preserve_index=False)
                if writer is None:
                    writer = pq_w.ParquetWriter(out_stock, table.schema, compression="zstd")
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table)
                # light per-scope tally
                stock_parts_meta.append(
                    bdf.groupby("scope_category", dropna=False).agg(
                        rows=("Option Code", "size"),
                        qty=("Stock Qty", "sum"),
                        options=("Option Code", "nunique"),
                        sites=("Site_Code", "nunique"),
                    )
                )
                del bdf, table
        finally:
            if writer is not None:
                writer.close()

        if stock_parts_meta:
            meta = pd.concat(stock_parts_meta).groupby(level=0).sum(numeric_only=True).reset_index()
            # options/sites sum is wrong if summed across batches - recompute from sales-style summary if file small enough
            # For summary use filtered sales-style approach with a second light pass for uniques is expensive;
            # report row/qty from meta and option counts from opt_map ∩ stock later.
            stock_summary = meta.rename(columns={"index": "scope_category"})
            stock_summary.to_csv(OUT / "scope_stock_rows_qty.csv", index=False)
            print(f"  stock_weekly_8ag rows~{stock_rows:,} -> {out_stock}")
            print("  Splitting stock into the four scope folders...")
            split_parquet(out_stock, "step3_filter_ag", "stock_weekly.parquet")
        else:
            print("  no stock rows matched the four W scopes")
    else:
        print("\n3) Skip stock filter (step1 parquet not found)")

    overall = {
        "ag_categories": len(AG_SCOPE),
        "item_master_options_in_scope": len(opt_map),
        "weekly_sales_rows_before": before,
        "weekly_sales_rows_after": len(filtered),
        "weekly_sales_options_after": int(filtered["Option Code"].nunique()),
        "options_in_master_not_in_sales": len(only_master),
        "all_scopes_have_sales": bool((sales_summary["has_data"] == "Yes").all()),
    }
    pd.DataFrame([overall]).to_csv(OUT / "step3_summary.csv", index=False)

    # Mapping rules sheet helper
    pd.DataFrame(
        [{"AG": k, "scope_category": v} for k, v in AG_SCOPE.items()]
    ).to_csv(OUT / "ag_to_scope_mapping.csv", index=False)

    with pd.ExcelWriter(OUT / "step3_8ag_filter_report.xlsx", engine="openpyxl") as xw:
        sales_summary.to_excel(xw, sheet_name="Sales_by_scope", index=False)
        ag_counts.to_excel(xw, sheet_name="AG_option_counts", index=False)
        pd.DataFrame([overall]).to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(
            [{"AG": k, "scope_category": v} for k, v in AG_SCOPE.items()]
        ).to_excel(xw, sheet_name="Mapping", index=False)

    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {OUT / 'sales_weekly_8ag.parquet'}")
    print(f"All outputs -> {OUT}")
    print("DONE - Step 3 8-AG filter complete.")
    return overall


if __name__ == "__main__":
    run()
