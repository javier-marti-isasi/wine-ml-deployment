import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.inference import WinePredictor
from app.schemas import PredictionResponse, WineInput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wine-api")


# Context manager para cargar el modelo al iniciar la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = WinePredictor.load()
    logger.info("Modelo cargado: %s", app.state.predictor.metadata["model_version"])
    yield

# Configuración de la aplicación FastAPI
app = FastAPI(
    title="Wine Quality ML API",
    version="1.0.0",
    description="Clasificación binaria: quality >= 6 sobre el dataset Wine Quality (UCI).",
    lifespan=lifespan,
)

# Endpoint de health check
@app.get("/health")
def health():
    return {"status": "ok"}


# Endpoint para obtener información del modelo
@app.get("/model-info")
def model_info(request: Request):
    return request.app.state.predictor.metadata


# Endpoint para hacer predicciones
@app.post("/predict", response_model=PredictionResponse)
def predict(data: WineInput, request: Request):
    start = time.perf_counter()
    result = request.app.state.predictor.predict(data)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    # Evitamos registrar los datos de entrada para no almacenar información sensible.
    logger.info("predict version=%s latency_ms=%.2f", result["model_version"], elapsed_ms)
    return result
