# src/balance_dataset.py

import argparse
from pathlib import Path

import pandas as pd


def main(input_csv: str, label_col: str, out_csv: str, random_state: int):
    df = pd.read_csv(input_csv)

    if label_col not in df.columns:
        raise ValueError(f"No existe la columna '{label_col}'")

    # Separar clases
    df_pos = df[df[label_col] == 1]
    df_neg = df[df[label_col] == 0]

    n_pos = len(df_pos)
    n_neg = len(df_neg)

    print(f"Clase 1 (positivos): {n_pos}")
    print(f"Clase 0 (negativos): {n_neg}")

    if n_pos == 0:
        raise ValueError("No hay positivos para balancear")

    # Undersampling de la clase mayoritaria
    df_neg_sampled = df_neg.sample(n=n_pos, random_state=random_state)

    df_balanced = pd.concat([df_pos, df_neg_sampled], axis=0)
    df_balanced = df_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    print("Dataset balanceado:", df_balanced.shape)
    print(df_balanced[label_col].value_counts())

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(out_csv, index=False)

    print(f"✅ Guardado en: {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default="data/processed/dataset_atchley.csv")
    p.add_argument("--label_col", default="is_sarscov2_related")
    p.add_argument("--out_csv", default="data/processed/dataset_atchley_balanced.csv")
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    main(
        input_csv=args.input_csv,
        label_col=args.label_col,
        out_csv=args.out_csv,
        random_state=args.random_state,
    )