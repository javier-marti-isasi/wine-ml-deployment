import numpy as np
import pytest
from pydantic import ValidationError
from unittest.mock import Mock

from app.inference import WinePredictor, features_to_array
from app.schemas import FEATURE_COLUMNS, WineInput


def test_input_valido(valid_wine):
    data = WineInput(**valid_wine)
    assert data.alcohol == 9.4


def test_valor_negativo_rechazado(valid_wine):
    valid_wine["alcohol"] = -1
    with pytest.raises(ValidationError):
        WineInput(**valid_wine)


def test_campos_desconocidos_rechazados(valid_wine):
    valid_wine["extra"] = 123
    with pytest.raises(ValidationError):
        WineInput(**valid_wine)


def test_orden_features(valid_wine):
    matrix = features_to_array(WineInput(**valid_wine))
    assert matrix.shape == (1, 11)
    assert np.isclose(matrix[0, FEATURE_COLUMNS.index("alcohol")], 9.4)


def test_inferencia_con_modelo_simulado(valid_wine):
    model = Mock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    predictor = WinePredictor(model, {"model_version": "test-v1"})
    result = predictor.predict(WineInput(**valid_wine))
    assert result == {
        "quality_good": True,
        "probability_good": 0.8,
        "model_version": "test-v1",
    }
    model.predict_proba.assert_called_once()
