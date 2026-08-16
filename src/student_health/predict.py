from collections.abc import Sequence
from typing import Any

import joblib
import pandas as pd

from student_health.config import (
    LABEL_ENCODER_PATH,
    MODEL_PATH,
)


class StudentHealthPredictor:
    def __init__(self) -> None:
        self.pipeline = joblib.load(MODEL_PATH)
        self.label_encoder = joblib.load(LABEL_ENCODER_PATH)

    def predict(
        self,
        records: Sequence[dict[str, Any]],
    ) -> list[str]:
        frame = pd.DataFrame(records)

        encoded_predictions = self.pipeline.predict(frame)

        predictions = self.label_encoder.inverse_transform(
            encoded_predictions.astype(int)
        )

        return predictions.tolist()

    def predict_proba(
        self,
        records: Sequence[dict[str, Any]],
    ) -> list[dict[str, float]]:
        frame = pd.DataFrame(records)
        probabilities = self.pipeline.predict_proba(frame)

        return [
            {
                class_name: float(probability)
                for class_name, probability in zip(
                    self.label_encoder.classes_,
                    row,
                )
            }
            for row in probabilities
        ]
