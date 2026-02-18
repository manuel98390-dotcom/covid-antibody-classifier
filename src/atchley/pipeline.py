# src/atchley/pipeline.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd

from .transform import aa_to_atchley_matrix, build_region_vector


REGIONS = ["FR1-IMGT", "CDR1-IMGT", "FR2-IMGT", "CDR2-IMGT", "FR3-IMGT"]
REQUIRED = ["Sequence ID"] + REGIONS

FEATURE_COLS = [
    "F1-FR1", "F2-FR1", "F3-FR1", "F4-FR1", "F5-FR1",
    "F1-CDR1", "F2-CDR1", "F3-CDR1", "F4-CDR1", "F5-CDR1",
    "F1-FR2", "F2-FR2", "F3-FR2", "F4-FR2", "F5-FR2",
    "F1-CDR2", "F2-CDR2", "F3-CDR2", "F4-CDR2", "F5-CDR2",
    "F1-FR3", "F2-FR3", "F3-FR3", "F4-FR3", "F5-FR3",
]


@dataclass
class BuildReport:
    n_input_heavy: int
    n_input_light: int
    n_heavy_clean: int
    n_light_clean: int
    n_pairs_merged: int
    n_pairs_valid: int
    n_pairs_dropped_invalid_aa: int
    heavy_nan_rows: int
    light_nan_rows: int
    heavy_star_rows: int
    light_star_rows: int
    heavy_dups: int
    light_dups: int


def _count_rows_with_star(df: pd.DataFrame, cols: List[str]) -> int:
    mask = df[cols].astype(str).apply(lambda s: s.str.contains(r"\*")).any(axis=1)
    return int(mask.sum())


def _remove_rows_with_star(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    mask = df[cols].astype(str).apply(lambda s: s.str.contains(r"\*")).any(axis=1)
    return df.loc[~mask].copy()


def _clean_imgt_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    df["Sequence ID"] = df["Sequence ID"].astype(str)

    # 1) NaN
    nan_rows = int(df[REQUIRED].isna().any(axis=1).sum())
    df = df.dropna(subset=REQUIRED).copy()

    # 2) '*'
    star_rows = _count_rows_with_star(df, REGIONS)
    df = _remove_rows_with_star(df, REGIONS)

    # 3) Duplicados por Sequence ID
    dup_count = int(df.duplicated(subset=["Sequence ID"]).sum())
    df = df.drop_duplicates("Sequence ID", keep="first").copy()

    stats: Dict[str, Any] = {
        "nan_rows": nan_rows,
        "star_rows": star_rows,
        "dup_count": dup_count,
        "n_clean": int(len(df)),
    }
    return df, stats


def _row_to_region_vector(row: pd.Series, suffix: str) -> Optional[List[float]]:
    fr1 = aa_to_atchley_matrix(row.get(f"FR1-IMGT{suffix}"))
    cdr1 = aa_to_atchley_matrix(row.get(f"CDR1-IMGT{suffix}"))
    fr2 = aa_to_atchley_matrix(row.get(f"FR2-IMGT{suffix}"))
    cdr2 = aa_to_atchley_matrix(row.get(f"CDR2-IMGT{suffix}"))
    fr3 = aa_to_atchley_matrix(row.get(f"FR3-IMGT{suffix}"))

    if None in (fr1, cdr1, fr2, cdr2, fr3):
        return None

    return build_region_vector(fr1, cdr1, fr2, cdr2, fr3)


def build_paired_atchley_features(
    heavy_csv: str,
    light_csv: str,
    output_csv: str,
    label_col: Optional[str] = None,
    label_value: Optional[int] = None,
) -> BuildReport:
    """
    Genera un dataset emparejado VH-VL con features Atchley:

    - Limpieza (NaN, '*', duplicados)
    - Merge por Sequence ID
    - Features Atchley para Heavy y Light
    - Output: un CSV con features (VH + VL) y etiqueta opcional

    Nota: las columnas de salida quedan con prefijo:
      - H_ para pesadas
      - L_ para ligeras
    """
    heavy = pd.read_csv(heavy_csv)
    light = pd.read_csv(light_csv)

    n_input_heavy, n_input_light = len(heavy), len(light)

    heavy_clean, sH = _clean_imgt_table(heavy)
    light_clean, sL = _clean_imgt_table(light)

    pairs = heavy_clean.merge(
        light_clean, on="Sequence ID", how="inner", suffixes=("_H", "_L")
    )
    n_pairs_merged = int(len(pairs))

    ids: List[str] = []
    Xh: List[List[float]] = []
    Xl: List[List[float]] = []
    dropped_invalid = 0

    for _, row in pairs.iterrows():
        vh = _row_to_region_vector(row, "_H")
        vl = _row_to_region_vector(row, "_L")
        if vh is None or vl is None:
            dropped_invalid += 1
            continue
        ids.append(row["Sequence ID"])
        Xh.append(vh)
        Xl.append(vl)

    df_h = pd.DataFrame(Xh, columns=[f"H_{c}" for c in FEATURE_COLS])
    df_l = pd.DataFrame(Xl, columns=[f"L_{c}" for c in FEATURE_COLS])

    out = pd.concat([df_h, df_l], axis=1)
    out.insert(0, "Sequence ID", ids)

    if label_col is not None and label_value is not None:
        out[label_col] = int(label_value)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    return BuildReport(
        n_input_heavy=n_input_heavy,
        n_input_light=n_input_light,
        n_heavy_clean=int(len(heavy_clean)),
        n_light_clean=int(len(light_clean)),
        n_pairs_merged=n_pairs_merged,
        n_pairs_valid=int(len(out)),
        n_pairs_dropped_invalid_aa=int(dropped_invalid),
        heavy_nan_rows=int(sH["nan_rows"]),
        light_nan_rows=int(sL["nan_rows"]),
        heavy_star_rows=int(sH["star_rows"]),
        light_star_rows=int(sL["star_rows"]),
        heavy_dups=int(sH["dup_count"]),
        light_dups=int(sL["dup_count"]),
    )