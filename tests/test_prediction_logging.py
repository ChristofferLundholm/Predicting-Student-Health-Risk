import json
import logging

from student_health import prediction_logging


def test_completed_prediction_log_is_structured_and_aggregated(
    monkeypatch,
):
    captured = []

    monkeypatch.setattr(
        prediction_logging.prediction_logger,
        "log",
        lambda level, message: captured.append((level, message)),
    )

    prediction_logging.log_prediction_completed(
        request_id="request-123",
        endpoint="/predict/batch",
        labels=["fit", "fit", "at-risk"],
        duration_ms=12.345,
    )

    level, message = captured[0]
    payload = json.loads(message)

    assert level == logging.INFO
    assert payload["event"] == "prediction_completed"
    assert payload["request_id"] == "request-123"
    assert payload["endpoint"] == "/predict/batch"
    assert payload["record_count"] == 3
    assert payload["prediction_counts"] == {
        "at-risk": 1,
        "fit": 2,
    }
    assert payload["duration_ms"] == 12.35
    assert "timestamp" in payload
    assert "records" not in payload
    assert "probabilities" not in payload


def test_failed_prediction_log_does_not_include_error_message(
    monkeypatch,
):
    captured = []

    monkeypatch.setattr(
        prediction_logging.prediction_logger,
        "log",
        lambda level, message: captured.append((level, message)),
    )

    prediction_logging.log_prediction_failed(
        request_id="request-456",
        endpoint="/predict",
        record_count=1,
        duration_ms=4.0,
        error_type="ValueError",
    )

    level, message = captured[0]
    payload = json.loads(message)

    assert level == logging.WARNING
    assert payload["event"] == "prediction_failed"
    assert payload["error_type"] == "ValueError"
    assert "error_message" not in payload
