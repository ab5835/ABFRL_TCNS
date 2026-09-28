"""Run the W D0 prep steps in pipeline order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STEPS = [
    "step1_stock_formatted.py",
    "step2_weekly_sales.py",
    "step3_filter_ag.py",
    "step4_merge_net_qty.py",
    "step4b_correct_mrp.py",
    "step5_skd_3n_filter.py",
    "step5a_weekly_option_discount.py",
    "step7_d0_weekly.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    for name in STEPS:
        script = here / name
        print("\n" + "=" * 70)
        print("RUN", name)
        print("=" * 70)
        result = subprocess.run([sys.executable, str(script)], cwd=str(here))
        if result.returncode != 0:
            raise SystemExit(f"{name} failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
