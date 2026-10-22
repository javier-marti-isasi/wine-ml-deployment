from pydantic import BaseModel, ConfigDict, Field


# El orden es compartido con entrenamiento e inferencia.
FEATURE_COLUMNS = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
    "density", "pH", "sulphates", "alcohol",
]


class WineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixed_acidity: float = Field(gt=0)
    volatile_acidity: float = Field(ge=0)
    citric_acid: float = Field(ge=0)
    residual_sugar: float = Field(ge=0)
    chlorides: float = Field(ge=0)
    free_sulfur_dioxide: float = Field(ge=0)
    total_sulfur_dioxide: float = Field(ge=0)
    density: float = Field(gt=0)
    pH: float = Field(ge=0, le=14)
    sulphates: float = Field(ge=0)
    alcohol: float = Field(gt=0)


class PredictionResponse(BaseModel):
    quality_good: bool
    probability_good: float = Field(ge=0, le=1)
    model_version: str
