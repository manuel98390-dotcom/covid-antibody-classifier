# Clasificador de Anticuerpos SARS-CoV-2 (VH–VL) usando Factores de Atchley

Este repositorio implementa un pipeline completo de aprendizaje automático de extremo a extremo (end-to-end) para la clasificación de anticuerpos relacionados con SARS-CoV-2, a partir de secuencias VH–VL exportadas desde IMGT.

El sistema abarca todo el flujo de trabajo:

- Procesamiento y limpieza de datos biológicos crudos.
- Emparejamiento estructurado de cadenas pesadas (VH) y ligeras (VL).
- Ingeniería de características mediante factores fisicoquímicos de Atchley.
- Construcción reproducible del dataset.
- Balanceo de clases en escenarios altamente desbalanceados.
- Entrenamiento y evaluación comparativa de múltiples modelos supervisados.

En la etapa de modelado se implementan y comparan los siguientes algoritmos:

- Regresión Logística, como baseline lineal interpretable.
- Support Vector Machine (SVM), para clasificación en espacios de alta dimensión.
- Random Forest, modelo de ensamble robusto ante no linealidades y ruido.

La selección del mejor modelo se realiza automáticamente utilizando la métrica F1-score, especialmente relevante debido al desbalance extremo presente en el dataset original (~3.5% clase positiva).

El proyecto demuestra el diseño modular de una arquitectura reproducible, desde datos crudos hasta modelo entrenado y métricas exportadas.

## Objetivo del proyecto

Construir un dataset estructurado y reproducible a partir de exportaciones IMGT, transformando secuencias de aminoácidos en vectores numéricos mediante factores de Atchley, para entrenar modelos capaces de distinguir anticuerpos relacionados con SARS-CoV-2.

## Colaboración institucional

Este trabajo forma parte de una colaboración técnica con el **Instituto Nacional de Salud Pública (INSP, México)**, enfocada en el análisis computacional de anticuerpos y la aplicación de técnicas de aprendizaje automático en contextos biomoleculares reales.

El desarrollo del pipeline refleja un enfoque reproducible y modular para el procesamiento de datos inmunológicos, desde secuencias crudas hasta modelos supervisados evaluados bajo métricas robustas.

## Fuentes de datos

- OAS (Observed Antibody Space) → anticuerpos no relacionados con SARS-CoV-2.
- CoV-AbDab (Coronavirus Antibody Database) → anticuerpos relacionados con SARS-CoV-2.

Los datos fueron procesados previamente en IMGT y exportados como archivos CSV independientes para cadenas VH y VL.

Más detalles técnicos en `DATASET.md`.

---

## Pipeline implementado

### 1️ Limpieza de datos IMGT
- Eliminación de valores NaN en regiones clave.
- Eliminación de secuencias con símbolo ``.
- Eliminación de IDs duplicados.
- Emparejamiento VH–VL por `Sequence ID`.

### 2️ Transformación a factores de Atchley
Cada región IMGT:
- FR1
- CDR1
- FR2
- CDR2
- FR3

Se transforma a una representación numérica de 5 dimensiones por aminoácido y se calcula el promedio regional.

Resultado:
- 25 features para VH
- 25 features para VL
- Total: 50 características por anticuerpo

### 3️ Construcción del dataset final
Se genera un CSV estructurado con:
- Features VH
- Features VL
- Etiqueta binaria `is_sarscov2_related`


## Cómo ejecutar el proyecto

### Construcción del dataset

### bash
python3 -m src.atchley.make_dataset \
  --cov_heavy "data/raw/cadenasPesadas/pesadasCovid.csv" \
  --cov_light "data/raw/cadenasLigeras/ligerasCovid.csv" \
  --oas_heavy "data/raw/cadenasPesadas/pesadasSanos.csv" \
  --oas_light "data/raw/cadenasLigeras/ligerasSanos.csv" \
  --out_csv "data/processed/dataset_atchley.csv"

 Entrenamiento con dataset original (desbalanceado)

### Entrenamiento con dataset original (desbalanceado)

python3 -m src.train \
  --input_csv "data/processed/dataset_atchley.csv" \
  --label_col "is_sarscov2_related" \
  --out_model "models/best_model.pkl" \
  --out_metrics "results/metrics.json"

### Balanceo de clases (submuestreo controlado)
python3 -m src.balance_dataset \
  --input_csv "data/processed/dataset_atchley.csv" \
  --label_col "is_sarscov2_related" \
  --out_csv "data/processed/dataset_atchley_balanced.csv"

### Resultados obtenidos

Distribución original:
	•	Clase 0 (no SARS-CoV-2): 71,891 (~96.5%)
	•	Clase 1 (SARS-CoV-2): 2,633 (~3.5%)

Dataset desbalanceado
	•	Mejor modelo: Random Forest
	•	F1 ≈ 0.30

Dataset balanceado
	•	Mejor modelo: Random Forest
	•	F1 ≈ 0.82
