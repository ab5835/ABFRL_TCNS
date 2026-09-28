"""
Relative paths for TCNS.

Layout, all inside this project:
  raw/        sales, stock, item master
  D0 prep/    scripts, plus output/ created when a step runs
  W/          W categories
  Aurelia/    Aurelia categories
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PREP_DIR = Path(__file__).resolve().parent
ROOT = PREP_DIR.parent
RAW = ROOT / "raw"
W = ROOT / "W"
AURELIA = ROOT / "Aurelia"

SALES_DIR = RAW / "Sale Data (1)"
SALES_PQ = RAW / "Sales_Data_TCNS_Merged_Daily.parquet"
STOCK_FOLDERS = [RAW / "Stock Data", RAW / "Stock Data (1)"]

_item_masters = sorted(RAW.glob("Item Master*.xlsb"))
ITEM_MASTER = _item_masters[0] if _item_masters else RAW / "Item Master  Barcode master 3rd-Sept-2026.xlsb"
ITEM_MASTER_FALLBACK_CSV = RAW / "item_master_style_scope_map.csv"

SCOPES = {
    "A|SKD|FESTIVE": AURELIA / "SKD" / "FESTIVE",
    "A|KURTA|FESTIVE": AURELIA / "KURTA" / "FESTIVE",
    "A|KURTA|NON FESTIVE": AURELIA / "KURTA" / "NON FESTIVE",
    "A|SKD|NON FESTIVE": AURELIA / "SKD" / "NON FESTIVE",
    "W|KURTA|FESTIVE": W / "KURTA" / "FESTIVE",
    "W|KURTA|NON FESTIVE": W / "KURTA" / "NON FESTIVE",
    "W|SKD|FESTIVE": W / "SKD" / "FESTIVE",
    "W|SKD|NON FESTIVE": W / "SKD" / "NON FESTIVE",
}

AG_SCOPE = {
    "A|SKD|FESTIVE": "Aurelia Festive 3 pc set",
    "A|KURTA|FESTIVE": "Aurelia Festive Kurta",
    "A|KURTA|NON FESTIVE": "Aurelia Non-Festive kurta",
    "A|SKD|NON FESTIVE": "Aurelia Non-Festive 3 pc set",
    "W|KURTA|FESTIVE": "W festive Kurta",
    "W|KURTA|NON FESTIVE": "W Non-Festive kurta",
    "W|SKD|FESTIVE": "W Festive 3 pc set",
    "W|SKD|NON FESTIVE": "W Non-Festive 3 pc set",
}
AG_ORDER = list(AG_SCOPE.keys())


def out_dir(step: str) -> Path:
    """Shared working file for the next step. Not the category result."""
    path = PREP_DIR / "output" / step
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scope_path(folder: Path, step: str, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{step}__{filename}"


def write_split(df: pd.DataFrame, step: str, filename: str, ag_col: str = "AG") -> None:
    """Write one scope's rows straight into that scope's results folder."""
    if df is None or ag_col not in getattr(df, "columns", []):
        print(f"  scope split skipped for {filename}: no column {ag_col}")
        return
    key = df[ag_col].astype("string").str.strip().str.upper()
    for ag, folder in SCOPES.items():
        part = df.loc[key == ag]
        out = _scope_path(folder, step, filename)
        if out.suffix == ".parquet":
            part.to_parquet(out, index=False)
        elif out.suffix == ".csv":
            part.to_csv(out, index=False)
        else:
            raise ValueError(f"unsupported split file type: {out.suffix}")
        print(f"  {ag}: {len(part):,} rows -> {out.relative_to(ROOT)}")


def split_parquet(src: Path, step: str, filename: str, ag_col: str = "AG") -> None:
    """Stream a large parquet into one file per scope results folder."""
    if src is None or not Path(src).exists():
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(src)
    writers: dict[str, pq.ParquetWriter] = {}
    counts = {ag: 0 for ag in SCOPES}
    try:
        for batch in pf.iter_batches(batch_size=400_000):
            frame = batch.to_pandas()
            if ag_col not in frame.columns:
                print(f"  scope split skipped for {Path(src).name}: no column {ag_col}")
                return
            key = frame[ag_col].astype("string").str.strip().str.upper()
            for ag, folder in SCOPES.items():
                part = frame.loc[key == ag]
                if len(part) == 0:
                    continue
                counts[ag] += len(part)
                out = _scope_path(folder, step, filename)
                table = pa.Table.from_pandas(part, preserve_index=False)
                if ag not in writers:
                    if out.exists():
                        out.unlink()
                    writers[ag] = pq.ParquetWriter(out, table.schema, compression="zstd")
                else:
                    table = table.cast(writers[ag].schema)
                writers[ag].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()
    for ag, n in counts.items():
        out = _scope_path(SCOPES[ag], step, filename)
        print(f"  {ag}: {n:,} rows -> {out.relative_to(ROOT)}")
