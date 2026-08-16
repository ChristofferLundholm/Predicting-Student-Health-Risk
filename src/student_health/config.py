import os
from pathlib import Path

RANDOM_STATE = 1

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(
    os.environ.get(
        "STUDENT_HEALTH_PROJECT_ROOT",
        str(DEFAULT_PROJECT_ROOT),
    )
)
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = ARTIFACT_DIR / "lightgbm_pipeline.joblib"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder.joblib"
