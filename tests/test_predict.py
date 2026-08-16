import pytest

from student_health.predict import StudentHealthPredictor


@pytest.fixture(scope="module")
def predictor():
    return StudentHealthPredictor()


@pytest.fixture
def student_record():
    return {
        "sleep_duration": 5.22,
        "heart_rate": 70.6,
        "bmi": 25.66,
        "calorie_expenditure": 2174.0,
        "step_count": 1326.0,
        "exercise_duration": 19.8,
        "water_intake": 1.86,
        "diet_type": "veg",
        "stress_level": "high",
        "sleep_quality": "average",
        "physical_activity_level": "sedentary",
        "smoking_alcohol": "yes",
        "gender": "female",
    }


def test_predicts_one_record(predictor, student_record):
    predictions = predictor.predict([student_record])

    assert len(predictions) == 1
    assert predictions[0] in {
        "at-risk",
        "fit",
        "unhealthy",
    }


def test_predict_proba_returns_all_classes(
    predictor,
    student_record,
):
    probabilities = predictor.predict_proba([student_record])

    assert len(probabilities) == 1

    result = probabilities[0]

    assert set(result) == {
        "at-risk",
        "fit",
        "unhealthy",
    }


def test_prediction_probabilities_are_floats(
    predictor,
    student_record,
):
    result = predictor.predict_proba([student_record])[0]

    assert all(
        isinstance(probability, float)
        for probability in result.values()
    )


def test_prediction_probabilities_sum_to_one(
    predictor,
    student_record,
):
    result = predictor.predict_proba([student_record])[0]

    assert sum(result.values()) == pytest.approx(1.0)