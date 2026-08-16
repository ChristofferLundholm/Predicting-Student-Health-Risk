import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from student_health.config import (
    ARTIFACT_DIR,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TRAIN_PATH,
)
from student_health.model_builder import (
    make_lightgbm_model,
    make_model_pipeline,
)


TARGET = "health_condition"
ID_COLUMN = "id"

LIGHTGBM_BEST_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.029906841779048114,
    "num_leaves": 43,
    "min_child_samples": 33,
    "min_split_gain": 1.2243833148888116e-08,
    "subsample": 0.7468435864736457,
    "colsample_bytree": 0.8868160240315067,
    "reg_alpha": 9.07360023662835e-06,
    "reg_lambda": 3.193708331097069e-05,
    "max_bin": 511,
    "cat_smooth": 2.2787612565628526,
    "cat_l2": 0.10846435804415694,
}


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)

    X = train_df.drop(columns=[TARGET, ID_COLUMN])
    y = train_df[TARGET]

    categorical_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    label_encoder = LabelEncoder()

    y_encoded = np.asarray(
        label_encoder.fit_transform(y),
        dtype=np.int64,
    )

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_encoded,
    )

    model = make_lightgbm_model(
        params=LIGHTGBM_BEST_PARAMS,
        random_state=RANDOM_STATE,
    )

    pipeline = make_model_pipeline(
        model=model,
        categorical_columns=categorical_columns,
    )

    pipeline.fit(
        X,
        y_encoded,
        model__sample_weight=sample_weights,
        model__categorical_feature=categorical_columns,
    )

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print(f"Saved model pipeline to {MODEL_PATH}")
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()
