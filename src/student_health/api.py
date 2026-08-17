from collections.abc import Sequence
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status

from student_health.predict import StudentHealthPredictor
from student_health.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    StudentHealthInput,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = StudentHealthPredictor()
    yield


app = FastAPI(
    title="Student Health Risk API",
    description="Predict a student's health condition from structured inputs.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_predictor(request: Request) -> StudentHealthPredictor:
    return request.app.state.predictor

PredictorDependency = Annotated[
    StudentHealthPredictor,
    Depends(get_predictor),
]

def make_predictions(
    records: Sequence[dict[str, Any]],
    predictor: StudentHealthPredictor,
) -> list[PredictionResponse]:
    try:
        labels = predictor.predict(records)
        probabilities = predictor.predict_proba(records)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error

    return [
        PredictionResponse(
            prediction=label,
            probabilities=record_probabilities,
        )
        for label, record_probabilities in zip(
            labels,
            probabilities,
            strict=True,
        )
    ]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    predictor: PredictorDependency,
) -> ModelInfoResponse:
    feature_names = [
        str(feature)
        for feature in getattr(
            predictor.pipeline,
            "feature_names_in_",
            [],
        )
    ]
    model = predictor.pipeline.named_steps["model"]

    return ModelInfoResponse(
        model_type=type(model).__name__,
        classes=[
            str(class_name)
            for class_name in predictor.label_encoder.classes_
        ],
        feature_count=len(feature_names),
        features=feature_names,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    record: StudentHealthInput,
    predictor: PredictorDependency,
) -> PredictionResponse:
    return make_predictions(
        records=[record.model_dump()],
        predictor=predictor,
    )[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    request: BatchPredictionRequest,
    predictor: PredictorDependency,
) -> BatchPredictionResponse:
    predictions = make_predictions(
        records=[
            record.model_dump()
            for record in request.records
        ],
        predictor=predictor,
    )
    return BatchPredictionResponse(predictions=predictions)
