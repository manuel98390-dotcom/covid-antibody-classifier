# src/train.py

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def eval_binary(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main(input_csv: str, label_col: str, out_model: str, out_metrics: str, test_size: float):
    df = pd.read_csv(input_csv)

    if label_col not in df.columns:
        raise ValueError(f"No existe la columna '{label_col}' en {input_csv}")

    y = df[label_col].astype(int)

    # Quitamos columnas no-features
    drop_cols = [label_col]
    for c in ["Sequence ID", "group", "source"]:
        if c in df.columns:
            drop_cols.append(c)

    X = df.drop(columns=drop_cols, errors="ignore")

    # Split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    # Modelos baseline
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=4000))
        ]),
        "linearsvm": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LinearSVC())
        ]),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }

    results = {}
    best_name = None
    best_f1 = -1.0
    best_model_obj = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        m = eval_binary(y_test, y_pred)
        results[name] = m

        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_name = name
            best_model_obj = model

    assert best_model_obj is not None

    # Guardar modelo
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_obj, out_model)

    # Métricas best model
    best_pred = best_model_obj.predict(X_test)
    report_best = classification_report(y_test, best_pred, output_dict=True, zero_division=0)

    payload = {
        "input_csv": input_csv,
        "label_col": label_col,
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "test_size": float(test_size),
        "label_distribution": df[label_col].value_counts().to_dict(),
        "best_model": best_name,
        "best_f1": float(best_f1),
        "per_model": results,
        "classification_report_best": report_best,
    }

    Path(out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("✅ Entrenamiento completado")
    print(f"Mejor modelo: {best_name} | F1={best_f1:.4f}")
    print(f"Modelo: {out_model}")
    print(f"Métricas: {out_metrics}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default="data/processed/dataset_atchley.csv")
    p.add_argument("--label_col", default="is_sarscov2_related")
    p.add_argument("--out_model", default="models/best_model.pkl")
    p.add_argument("--out_metrics", default="results/metrics.json")
    p.add_argument("--test_size", type=float, default=0.2)
    args = p.parse_args()

    main(
        input_csv=args.input_csv,
        label_col=args.label_col,
        out_model=args.out_model,
        out_metrics=args.out_metrics,
        test_size=args.test_size,
    )