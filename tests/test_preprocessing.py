import pandas as pd
import pytest
from student_health.preprocessing import NativeCategoryTransformer

def test_fit_rejects_missing_categorical_column():
    frame = pd.DataFrame(
        {
            "bmi": [22.0, 25.0],
            "gender": ["female", "male"],
        }
    )

    transformer = NativeCategoryTransformer(
        columns=["gender", "diet_type"]
    )

    with pytest.raises(ValueError) as error:
        transformer.fit(frame)

    assert "Missing required columns" in str(error.value)
    assert "diet_type" in str(error.value)

def test_transform_rejects_missing_input_column():
    training_frame = pd.DataFrame(
        {
            "bmi": [22.0, 25.0],
            "gender": ["female", "male"],
            "diet_type": ["veg", "non-veg"],
        }
    )

    transformer = NativeCategoryTransformer(
        columns=["gender", "diet_type"]
    )

    transformer.fit(training_frame)

    incomplete_frame = pd.DataFrame(
        {
            "gender": ["female"],
            "diet_type": ["veg"],
        }
    )

    with pytest.raises(ValueError) as error:
        transformer.transform(incomplete_frame)

    assert "Missing required columns" in str(error.value)
    assert "bmi" in str(error.value)

def test_transform_discards_unexpected_columns():
    training_frame = pd.DataFrame(
        {
            "bmi": [22.0, 25.0],
            "gender": ["female", "male"],
            "diet_type": ["veg", "non-veg"],
        }
    )

    transformer = NativeCategoryTransformer(
        columns=["gender", "diet_type"]
    )
    transformer.fit(training_frame)

    inference_frame = pd.DataFrame(
        {
            "diet_type": ["veg"],
            "unexpected_field": ["ignored"],
            "gender": ["female"],
            "bmi": [23.0],
        }
    )

    transformed = transformer.transform(inference_frame)

    assert transformed.columns.tolist() == [
        "bmi",
        "gender",
        "diet_type",
    ]
    assert "unexpected_field" not in transformed.columns

def test_missing_category_uses_missing_sentinel():
    frame = pd.DataFrame(
        {
            "bmi": [22.0, 25.0],
            "gender": ["female", None],
        }
    )

    transformer = NativeCategoryTransformer(
        columns=["gender"]
    )

    transformed = transformer.fit_transform(frame)

    assert transformed["gender"].tolist() == [
        "female",
        "__MISSING__",
    ]
    assert "__MISSING__" in (
        transformed["gender"].cat.categories
        )


def test_transform_reuses_training_categories():
    training_frame = pd.DataFrame(
        {
            "gender": ["female", "male"],
        }
    )

    inference_frame = pd.DataFrame(
        {
            "gender": ["male"],
        }
    )

    transformer = NativeCategoryTransformer(
        columns=["gender"]
    )

    transformer.fit(training_frame)
    transformed = transformer.transform(inference_frame)

    assert transformed["gender"].cat.categories.tolist() == [
        "female",
        "male",
    ]