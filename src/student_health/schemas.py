from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StudentHealthInput(BaseModel):
    """Raw features expected by the trained model."""

    model_config = ConfigDict(extra="forbid")

    sleep_duration: float | None
    heart_rate: float | None
    bmi: float | None
    calorie_expenditure: float | None
    step_count: float | None
    exercise_duration: float | None
    water_intake: float | None
    diet_type: str | None
    stress_level: str | None
    sleep_quality: str | None
    physical_activity_level: str | None
    smoking_alcohol: str | None
    gender: str | None


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[StudentHealthInput] = Field(
        min_length=1,
        max_length=100,
    )


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ModelInfoResponse(BaseModel):
    model_type: str
    classes: list[str]
    feature_count: int
    features: list[str]
