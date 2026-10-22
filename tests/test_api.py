import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    # El context manager activa el lifespan y carga el modelo real.
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["model_type"] == "CatBoostClassifier"
    assert len(response.json()["features"]) == 11


def test_predict_ok(client, valid_wine):
    response = client.post("/predict", json=valid_wine)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result["quality_good"], bool)
    assert 0 <= result["probability_good"] <= 1
    assert len(result["model_version"]) == 12


def test_predict_422(client, valid_wine):
    valid_wine["fixed_acidity"] = -5
    response = client.post("/predict", json=valid_wine)
    assert response.status_code == 422


def test_predict_falta_campo(client, valid_wine):
    del valid_wine["alcohol"]
    assert client.post("/predict", json=valid_wine).status_code == 422
