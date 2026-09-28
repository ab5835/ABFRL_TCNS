"""
TCNS D0 — Step 1: Format stock into join-ready weekly long table.

Raw TCNS stock is already long (one CSV/XLSX per snapshot date), unlike Tasva wide SOH.
This step:
1. Index Stock Data + Stock Data (1) (csv + xlsx)
2. Deduplicate by snapshot date (prefer CSV, then Stock Data (1))
3. Normalize CSV vs XLSX schemas
4. Add Date / Week_End_Date / Year / Month / Week / Week_Start
5. Align column names for downstream sales merge
6. Integrity + calendar checks
7. Save parquet + summary CSVs under step1_stock_formatted/output/
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import sys

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
from paths import STOCK_FOLDERS, out_dir

FOLDERS = STOCK_FOLDERS
OUT = out_dir("step1_stock_formatted")

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

CANON_COLS = [
    "Date",
    "Week_End_Date",
    "Week_Start",
    "Year",
    "Month",
    "Week",
    "Site_Code",
    "Option Code",
    "Brand",
    "Category",
    "Final_Cat",
    "Season",
    "Season_rmks",
    "Stock Qty",
    "Pivotal",
    "Non_pivotal",
    "available_sizes",
    "Validity",
    "source_file",
    "source_folder",
]


def parse_date(path: Path) -> pd.Timestamp | None:
    m = DATE_RE.search(path.name)
    if not m:
        return None
    return pd.Timestamp(m.group(1))


def index_snapshots() -> dict[pd.Timestamp, Path]:
    """One file per snapshot date. Prefer CSV over XLSX; prefer Stock Data (1)."""
    candidates: dict[pd.Timestamp, list[Path]] = {}
    for folder in FOLDERS:
        if not folder.exists():
            print("WARN missing folder:", folder)
            continue
        for path in list(folder.glob("*.csv")) + list(folder.glob("*.xlsx")):
            d = parse_date(path)
            if d is None:
                print("SKIP (no date):", path)
                continue
            candidates.setdefault(d, []).append(path)

    chosen: dict[pd.Timestamp, Path] = {}
    for d, paths in sorted(candidates.items()):

        def rank(p: Path):
            is_csv = 0 if p.suffix.lower() == ".csv" else 1
            folder_pref = 0 if p.parent.name == "Stock Data (1)" else 1
            return (is_csv, folder_pref, p.name)

        best = sorted(paths, key=rank)[0]
        chosen[d] = best
        if len(paths) > 1:
            print(
                f"  date {d.date()}: keep {best.parent.name}/{best.name}; "
                f"skip {len(paths) - 1} dup(s)"
            )
    return chosen


def _as_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def enrich(df: pd.DataFrame, snap: pd.Timestamp, path: Path) -> pd.DataFrame:
    week_start = snap - pd.Timedelta(days=int(snap.dayofweek))  # Monday
    out = df.copy()
    out["Date"] = snap
    out["Week_End_Date"] = snap  # snapshot date used as week anchor
    out["Week_Start"] = week_start.normalize()
    out["Year"] = int(snap.year)
    out["Month"] = int(snap.month)
    out["Week"] = int(snap.isocalendar().week)
    out["Stock Qty"] = pd.to_numeric(out.get("Stock Qty"), errors="coerce")
    out["Pivotal"] = pd.to_numeric(out.get("Pivotal"), errors="coerce")
    out["Non_pivotal"] = pd.to_numeric(out.get("Non_pivotal"), errors="coerce")
    if "Season_rmks" not in out.columns:
        out["Season_rmks"] = pd.NA
    if "Final_Cat" not in out.columns:
        out["Final_Cat"] = pd.NA
    out["source_file"] = path.name
    out["source_folder"] = path.parent.name
    for c in [
        "Site_Code",
        "Option Code",
        "Brand",
        "Category",
        "Final_Cat",
        "Season",
        "Season_rmks",
        "available_sizes",
        "Validity",
        "source_file",
        "source_folder",
    ]:
        out[c] = _as_str(out[c])
    for c in CANON_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[CANON_COLS]


def normalize_csv(df: pd.DataFrame, snap: pd.Timestamp, path: Path) -> pd.DataFrame:
    out = df.rename(
        columns={
            "style_code": "Option Code",
            "QTY": "Stock Qty",
        }
    ).copy()
    if "Category" in out.columns:
        out["Category"] = _as_str(out["Category"]).str.upper()
    out["Final_Cat"] = pd.NA
    out["Season_rmks"] = pd.NA
    return enrich(out, snap, path)


def normalize_xlsx(df: pd.DataFrame, snap: pd.Timestamp, path: Path) -> pd.DataFrame:
    out = df.rename(
        columns={
            "STOREID": "Site_Code",
            "ITEMID": "Option Code",
            "Cat": "Category",
            "Final_Cat": "Final_Cat",
            "Season rmks": "Season_rmks",
            "Qty": "Stock Qty",
        }
    ).copy()
    if "Category" in out.columns:
        out["Category"] = _as_str(out["Category"]).str.upper()
    if "Final_Cat" in out.columns:
        out["Final_Cat"] = _as_str(out["Final_Cat"]).str.upper()
    return enrich(out, snap, path)


def read_one(path: Path, snap: pd.Timestamp) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, low_memory=False)
        return normalize_csv(df, snap, path)
    try:
        df = pd.read_excel(path, sheet_name=0, engine="calamine")
    except Exception:
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    return normalize_xlsx(df, snap, path)


def calendar_checks(dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(dates):
        gap = (d - dates[i - 1]).days if i else None
        rows.append(
            {
                "Date": d.date(),
                "weekday": d.day_name(),
                "gap_from_prev_days": gap,
                "Year": d.year,
                "Month": d.month,
                "Week": int(d.isocalendar().week),
                "Week_Start": (d - pd.Timedelta(days=int(d.dayofweek))).date(),
            }
        )
    return pd.DataFrame(rows)


def integrity_summary(stock: pd.DataFrame, cov: pd.DataFrame) -> dict:
    dates = sorted(stock["Date"].dropna().dt.normalize().unique())
    gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))] if len(dates) > 1 else []
    irregular = [g for g in gaps if g != 7]
    expected_mon = pd.date_range(dates[0], dates[-1], freq="W-MON") if dates else []
    present = set(pd.to_datetime(dates))
    missing_mon = [d for d in expected_mon if d not in present]
    return {
        "stock_rows": len(stock),
        "nonzero_rows": int((stock["Stock Qty"].fillna(0) != 0).sum()),
        "total_stock_qty": float(stock["Stock Qty"].sum(skipna=True)),
        "unique_sites": int(stock["Site_Code"].nunique()),
        "unique_options": int(stock["Option Code"].nunique()),
        "unique_brands": int(stock["Brand"].nunique()),
        "unique_categories": int(stock["Category"].nunique()),
        "snapshots": len(dates),
        "date_min": str(dates[0].date()) if dates else None,
        "date_max": str(dates[-1].date()) if dates else None,
        "median_gap_days": float(np.median(gaps)) if gaps else None,
        "irregular_gap_count": len(irregular),
        "missing_monday_count": len(missing_mon),
        "missing_mondays": "; ".join(str(d.date()) for d in missing_mon[:20]),
        "null_date": int(stock["Date"].isna().sum()),
        "null_site": int((stock["Site_Code"].isna() | (stock["Site_Code"] == "")).sum()),
        "null_option": int(
            (stock["Option Code"].isna() | (stock["Option Code"] == "")).sum()
        ),
        "null_qty": int(stock["Stock Qty"].isna().sum()),
        "all_ok_keys": int(
            stock[["Date", "Site_Code", "Option Code"]].isna().sum().sum()
        )
        == 0
        and int((stock["Site_Code"] == "").sum()) == 0
        and int((stock["Option Code"] == "").sum()) == 0,
    }


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Stock folders:", FOLDERS)
    print("Output:", OUT)

    print("\nIndexing stock snapshots...")
    snaps = index_snapshots()
    dates = sorted(snaps)
    print(f"Unique snapshots: {len(dates)}")
    if not dates:
        raise FileNotFoundError(f"No stock files found under {FOLDERS}")
    print(f"Range: {dates[0].date()} -> {dates[-1].date()}")

    cov = calendar_checks(dates)
    cov["file"] = [snaps[d].name for d in dates]
    cov["folder"] = [snaps[d].parent.name for d in dates]
    cov["ext"] = [snaps[d].suffix.lower() for d in dates]
    cov.to_csv(OUT / "stock_week_calendar.csv", index=False)

    irregular = cov[cov["gap_from_prev_days"].notna() & (cov["gap_from_prev_days"] != 7)]
    print("Non-7-day gaps:", len(irregular))
    if len(irregular):
        print(irregular[["Date", "gap_from_prev_days", "file"]].to_string(index=False))

    # Merge to parquet (skip rebuild if a complete file already exists)
    out_parquet = OUT / "stock_weekly_long.parquet"
    rebuild = True
    if out_parquet.exists() and out_parquet.stat().st_size > 50_000_000:
        try:
            n_existing = pq.ParquetFile(out_parquet).metadata.num_rows
            if n_existing > 1_000_000:
                print(
                    f"\nExisting parquet found ({n_existing:,} rows, "
                    f"{out_parquet.stat().st_size/1e6:.1f} MB) — skipping rebuild."
                )
                rebuild = False
                total_rows = n_existing
                per_date = []
                sample_parts = []
        except Exception as e:
            print("Could not read existing parquet, will rebuild:", e)

    if rebuild:
        if out_parquet.exists():
            out_parquet.unlink()

        writer = None
        total_rows = 0
        per_date = []
        sample_parts = []

        try:
            for i, d in enumerate(dates, start=1):
                path = snaps[d]
                print(f"[{i}/{len(dates)}] {d.date()} <- {path.parent.name}/{path.name}")
                df = read_one(path, d)
                n = len(df)
                total_rows += n
                per_date.append(
                    {
                        "Date": d.date(),
                        "rows": n,
                        "sites": df["Site_Code"].nunique(),
                        "options": df["Option Code"].nunique(),
                        "stock_qty": float(df["Stock Qty"].sum(skipna=True)),
                        "file": path.name,
                        "folder": path.parent.name,
                    }
                )
                if i <= 2:
                    sample_parts.append(df.head(100))
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_parquet, table.schema, compression="zstd")
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table)
                del df, table
        finally:
            if writer is not None:
                writer.close()

        rows_df = pd.DataFrame(per_date)
        rows_df.to_csv(OUT / "stock_rows_per_snapshot.csv", index=False)
        if sample_parts:
            pd.concat(sample_parts, ignore_index=True).to_csv(
                OUT / "stock_weekly_long_sample.csv", index=False
            )
    else:
        # Ensure calendar/rows files exist from this run's index
        if not (OUT / "stock_rows_per_snapshot.csv").exists():
            # lightweight per-date row counts from parquet Date column
            pf0 = pq.ParquetFile(out_parquet)
            date_counts: dict[str, int] = {}
            for batch in pf0.iter_batches(batch_size=500_000, columns=["Date"]):
                bdf = batch.to_pandas()
                vc = bdf["Date"].dt.normalize().value_counts()
                for k, v in vc.items():
                    key = str(pd.Timestamp(k).date())
                    date_counts[key] = date_counts.get(key, 0) + int(v)
            pd.DataFrame(
                [{"Date": k, "rows": v, "file": snaps[pd.Timestamp(k)].name if pd.Timestamp(k) in snaps else ""}
                 for k, v in sorted(date_counts.items())]
            ).to_csv(OUT / "stock_rows_per_snapshot.csv", index=False)

    print("\nBuilding integrity summary (without loading full table into pandas)...")
    # Light scan: only key numeric/date columns via pyarrow to avoid RAM blow-up
    pf = pq.ParquetFile(out_parquet)
    total_qty = 0.0
    null_qty = 0
    null_date = 0
    null_site = 0
    null_option = 0
    nonzero = 0
    sites: set[str] = set()
    options: set[str] = set()
    brands: set[str] = set()
    cats: set[str] = set()

    for batch in pf.iter_batches(
        batch_size=500_000,
        columns=[
            "Date",
            "Site_Code",
            "Option Code",
            "Brand",
            "Category",
            "Stock Qty",
        ],
    ):
        bdf = batch.to_pandas()
        qty = pd.to_numeric(bdf["Stock Qty"], errors="coerce")
        total_qty += float(qty.fillna(0).sum())
        null_qty += int(qty.isna().sum())
        nonzero += int((qty.fillna(0) != 0).sum())
        null_date += int(bdf["Date"].isna().sum())
        site = bdf["Site_Code"].astype(str)
        opt = bdf["Option Code"].astype(str)
        null_site += int(((site == "") | (site == "nan") | (site == "<NA>")).sum())
        null_option += int(((opt == "") | (opt == "nan") | (opt == "<NA>")).sum())
        sites.update(site.unique().tolist())
        options.update(opt.unique().tolist())
        brands.update(bdf["Brand"].astype(str).unique().tolist())
        cats.update(bdf["Category"].astype(str).unique().tolist())

    sites.discard("nan")
    sites.discard("<NA>")
    options.discard("nan")
    options.discard("<NA>")
    brands.discard("nan")
    brands.discard("<NA>")
    cats.discard("nan")
    cats.discard("<NA>")

    dates_ts = sorted(pd.to_datetime(dates))
    gaps = [
        (dates_ts[i] - dates_ts[i - 1]).days for i in range(1, len(dates_ts))
    ]
    irregular_gaps = [g for g in gaps if g != 7]
    expected_mon = pd.date_range(dates_ts[0], dates_ts[-1], freq="W-MON")
    present = set(pd.to_datetime([d.normalize() for d in dates_ts]))
    missing_mon = [d for d in expected_mon if d not in present]

    summary = {
        "stock_rows": total_rows,
        "nonzero_rows": nonzero,
        "total_stock_qty": total_qty,
        "unique_sites": len(sites),
        "unique_options": len(options),
        "unique_brands": len(brands),
        "unique_categories": len(cats),
        "snapshots": len(dates_ts),
        "date_min": str(dates_ts[0].date()),
        "date_max": str(dates_ts[-1].date()),
        "median_gap_days": float(np.median(gaps)) if gaps else None,
        "irregular_gap_count": len(irregular_gaps),
        "missing_monday_count": len(missing_mon),
        "missing_mondays": "; ".join(str(d.date()) for d in missing_mon[:20]),
        "null_date": null_date,
        "null_site": null_site,
        "null_option": null_option,
        "null_qty": null_qty,
        "all_ok_keys": null_date == 0 and null_site == 0 and null_option == 0,
        "parquet_mb": round(out_parquet.stat().st_size / 1e6, 1),
    }
    pd.DataFrame([summary]).to_csv(OUT / "stock_integrity_checks.csv", index=False)

    # Non-zero parquet via pyarrow filter (stream write)
    print("Writing nonzero parquet...")
    nz_path = OUT / "stock_weekly_long_nonzero.parquet"
    if nz_path.exists():
        nz_path.unlink()
    nz_writer = None
    nz_rows = 0
    try:
        for batch in pf.iter_batches(batch_size=500_000):
            bdf = batch.to_pandas()
            bdf = bdf.loc[pd.to_numeric(bdf["Stock Qty"], errors="coerce").fillna(0) != 0]
            if len(bdf) == 0:
                continue
            nz_rows += len(bdf)
            table = pa.Table.from_pandas(bdf, preserve_index=False)
            if nz_writer is None:
                nz_writer = pq.ParquetWriter(nz_path, table.schema, compression="zstd")
            else:
                table = table.cast(nz_writer.schema)
            nz_writer.write_table(table)
    finally:
        if nz_writer is not None:
            nz_writer.close()

    # Brand / category overview from last batches already tracked
    pd.DataFrame({"Brand": sorted(brands)}).assign(note="unique brands seen").to_csv(
        OUT / "stock_brands_list.csv", index=False
    )
    pd.DataFrame({"Category": sorted(cats)}).assign(note="unique categories seen").to_csv(
        OUT / "stock_categories_list.csv", index=False
    )

    # Row counts by brand/category from a second light scan
    brand_rows: dict[str, int] = {}
    cat_rows: dict[str, int] = {}
    for batch in pf.iter_batches(batch_size=500_000, columns=["Brand", "Category"]):
        bdf = batch.to_pandas()
        for k, v in bdf["Brand"].astype(str).value_counts().items():
            brand_rows[k] = brand_rows.get(k, 0) + int(v)
        for k, v in bdf["Category"].astype(str).value_counts().items():
            cat_rows[k] = cat_rows.get(k, 0) + int(v)
    pd.DataFrame(
        [{"Brand": k, "rows": v} for k, v in sorted(brand_rows.items(), key=lambda x: -x[1])]
    ).to_csv(OUT / "stock_by_brand.csv", index=False)
    pd.DataFrame(
        [{"Category": k, "rows": v} for k, v in sorted(cat_rows.items(), key=lambda x: -x[1])]
    ).to_csv(OUT / "stock_by_category.csv", index=False)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {out_parquet}  ({out_parquet.stat().st_size/1e6:.1f} MB)")
    print(f"Wrote {nz_path}  rows={nz_rows:,}  ({nz_path.stat().st_size/1e6:.1f} MB)")
    print(f"All outputs -> {OUT}")
    print("DONE — stock formatted.")
    return summary


if __name__ == "__main__":
    run()
