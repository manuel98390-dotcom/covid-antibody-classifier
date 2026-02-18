# src/make_dataset.py

import argparse
import json
from pathlib import Path

import pandas as pd

from src.atchley.pipeline import build_paired_atchley_features


def main(
    cov_heavy: str,
    cov_light: str,
    oas_heavy: str,
    oas_light: str,
    out_csv: str,
):
    out_dir = Path(out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cov_out = out_dir / "covabdab_paired_atchley.csv"
    oas_out = out_dir / "oas_paired_atchley.csv"

    rep_cov = build_paired_atchley_features(
        heavy_csv=cov_heavy,
        light_csv=cov_light,
        output_csv=str(cov_out),
        label_col="is_sarscov2_related",
        label_value=1,
    )

    rep_oas = build_paired_atchley_features(
        heavy_csv=oas_heavy,
        light_csv=oas_light,
        output_csv=str(oas_out),
        label_col="is_sarscov2_related",
        label_value=0,
    )

    df_cov = pd.read_csv(cov_out)
    df_oas = pd.read_csv(oas_out)

    final = pd.concat([df_cov, df_oas], axis=0, ignore_index=True)
    final.to_csv(out_csv, index=False)

    report = {
        "covabdab": rep_cov.__dict__,
        "oas": rep_oas.__dict__,
        "final_rows": int(len(final)),
        "final_cols": int(final.shape[1]),
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/dataset_build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("✅ Dataset construido")
    print(f" - Output: {out_csv}")
    print(" - Reporte: results/dataset_build_report.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cov_heavy", required=True)
    p.add_argument("--cov_light", required=True)
    p.add_argument("--oas_heavy", required=True)
    p.add_argument("--oas_light", required=True)
    p.add_argument("--out_csv", default="data/processed/dataset_atchley.csv")
    args = p.parse_args()

    main(
        cov_heavy=args.cov_heavy,
        cov_light=args.cov_light,
        oas_heavy=args.oas_heavy,
        oas_light=args.oas_light,
        out_csv=args.out_csv,
    )