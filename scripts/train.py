"""Descarga Wine Quality UCI, prepara datos con Polars Lazy y entrena CatBoost.

Ejecutar desde la raíz: python -m scripts.train
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import polars as pl
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

from app.schemas import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "winequality-red.csv"
ARTIFACT_DIR = ROOT / "artifacts"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def main():
    # 1. Descargar dataset
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        print("Descargando Wine Quality desde UCI...")
        urlretrieve(URL, CSV_PATH)

    # 2. Preprocesamiento con Polars Lazy. Las operaciones se ejecutan al llamar a collect().
    query = (
        pl.scan_csv(CSV_PATH, separator=";", infer_schema_length=None)
        .rename({name.replace("_", " "): name for name in FEATURE_COLUMNS if "_" in name})
        .select([*FEATURE_COLUMNS, "quality"])
        .drop_nulls()
        .with_columns((pl.col("quality") >= 6).cast(pl.Int8).alias("target"))
    )

    print("Plan optimizado de Polars:")
    print(query.explain(optimized=True))
    df = query.collect()

    # 3. Separar features y target
    X = df.select(FEATURE_COLUMNS).to_numpy()
    y = df["target"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Entrenar modelo CatBoost
    model = CatBoostClassifier(
        iterations=150,
        depth=5,
        learning_rate=0.08,
        loss_function="Logloss",
        random_seed=42,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)

    # 5. Evaluar modelo
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # 6. Guardar modelo y metadatos
    model_path = ARTIFACT_DIR / "model.cbm"
    model.save_model(str(model_path))

    version = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12] # Generar hash del modelo
    metadata = {
        "model_version": version,
        "model_type": "CatBoostClassifier",
        "dataset": "UCI Wine Quality - red wine",
        "target_definition": "quality >= 6",
        "features": FEATURE_COLUMNS,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "r2": round(float(r2_score(y_test, proba)), 4),
        },
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Artefactos guardados en", ARTIFACT_DIR)
    print("Métricas:", metadata["metrics"])
    print("Versión:", version)


if __name__ == "__main__":
    main()
