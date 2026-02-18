# src/atchley/transform.py

import statistics
import pandas as pd
from .factors import FACTORES_ATCHLEY, VALID_AA


def aa_to_atchley_matrix(seq: str):
    if seq is None or pd.isna(seq):
        return None

    seq = str(seq).strip().upper()

    if not seq:
        return None

    if any(aa not in VALID_AA for aa in seq):
        return None

    return [FACTORES_ATCHLEY[aa] for aa in seq]


def mean_5d(matrix_lx5):
    return [
        round(statistics.mean([vec[i] for vec in matrix_lx5]), 3)
        for i in range(5)
    ]


def build_region_vector(fr1, cdr1, fr2, cdr2, fr3):
    return (
        mean_5d(fr1) +
        mean_5d(cdr1) +
        mean_5d(fr2) +
        mean_5d(cdr2) +
        mean_5d(fr3)
    )