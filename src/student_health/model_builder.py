from collections.abc import Mapping, Sequence
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from student_health.preprocessing import make_preprocessing_pipeline


def make_lightgbm_model(
    params: Mapping[str, Any],
    random_state: int,
) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=388,
        subsample_freq=1,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )


def make_model_pipeline(
    model: LGBMClassifier,
    categorical_columns: Sequence[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessing",
                make_preprocessing_pipeline(categorical_columns),
            ),
            ("model", model),
        ]
    )