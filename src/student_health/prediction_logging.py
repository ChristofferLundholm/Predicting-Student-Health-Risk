import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone

prediction_logger = logging.getLogger("student_health.predictions")
prediction_logger.setLevel(logging.INFO)
prediction_logger.propagate = False

if not prediction_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    prediction_logger.addHandler(handler)


def _write_event(level: int, event: dict[str, object]) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    prediction_logger.log(
        level,
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )


def log_prediction_completed(
    *,
    request_id: str,
    endpoint: str,
    labels: list[str],
    duration_ms: float,
) -> None:
    _write_event(
        logging.INFO,
        {
            "event": "prediction_completed",
            "request_id": request_id,
            "endpoint": endpoint,
            "record_count": len(labels),
            "prediction_counts": dict(Counter(labels)),
            "duration_ms": round(duration_ms, 2),
        },
    )


def log_prediction_failed(
    *,
    request_id: str,
    endpoint: str,
    record_count: int,
    duration_ms: float,
    error_type: str,
) -> None:
    _write_event(
        logging.WARNING,
        {
            "event": "prediction_failed",
            "request_id": request_id,
            "endpoint": endpoint,
            "record_count": record_count,
            "duration_ms": round(duration_ms, 2),
            "error_type": error_type,
        },
    )
