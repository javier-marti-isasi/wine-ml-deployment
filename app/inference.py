import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from app.schemas import FEATURE_COLUMNS, WineInput


ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def features_to_array(data: WineInput) -> np.ndarray:
    """Construye una matriz (1, 11) en el mismo orden del entrenamiento."""
    values = data.model_dump()
    return np.array([[values[name] for name in FEATURE_COLUMNS]], dtype=np.float64)


class WinePredictor:
    def __init__(self, model: CatBoostClassifier, metadata: dict):
        self.model = model
        self.metadata = metadata

    @classmethod
    def load(cls, directory: Path = ARTIFACT_DIR) -> "WinePredictor":
        model_path = directory / "model.cbm"
        metadata_path = directory / "metadata.json"
        if not model_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("Faltan artefactos. Ejecuta: python -m scripts.train")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["features"] != FEATURE_COLUMNS:
            raise RuntimeError("El orden de features del artefacto no coincide con la API")
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return cls(model, metadata)

    def predict(self, data: WineInput) -> dict:
        probability = float(self.model.predict_proba(features_to_array(data))[0, 1])
        return {
            "quality_good": probability >= 0.5,
            "probability_good": probability,
            "model_version": self.metadata["model_version"],
        }
