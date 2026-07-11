"""Compatibility wrappers for the engineering-layer math functions."""

from .state_machine import append_behavioral_confidence


def is_inside_polygon(*args, **kwargs):
    from .detection import is_inside_polygon as _is_inside_polygon

    return _is_inside_polygon(*args, **kwargs)


def process_abandoned_logic(*args, **kwargs):
    from .detection import process_abandoned_logic as _process_abandoned_logic

    return _process_abandoned_logic(*args, **kwargs)


def calculate_normalized_kinematics(*args, **kwargs):
    from .detection import calculate_normalized_kinematics as _calculate_normalized_kinematics

    return _calculate_normalized_kinematics(*args, **kwargs)


def verify_panic_anomaly(*args, **kwargs):
    from .detection import verify_panic_anomaly as _verify_panic_anomaly

    return _verify_panic_anomaly(*args, **kwargs)


def update_behavioral_filter(person_id: int, confidence: float, max_window: int = 15) -> None:
    """Append a confidence sample to the rolling behavioral buffer."""
    append_behavioral_confidence(person_id, confidence, max_window=max_window)


__all__ = [
    "is_inside_polygon",
    "process_abandoned_logic",
    "calculate_normalized_kinematics",
    "verify_panic_anomaly",
    "update_behavioral_filter",
]
