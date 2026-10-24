"""Opt-in telemetry for local gameplay; profile/DDA policy is a separate stage."""

from .recorder import TelemetryRecorder
from .schema import validate_event
from .storage import read_session

__all__ = ["TelemetryRecorder", "read_session", "validate_event"]
