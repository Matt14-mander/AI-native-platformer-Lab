"""Constant-space physical-tick motion facts; no heuristic skill labels."""

from math import isfinite

MOTION_VERSION = 1
STATIONARY_EPSILON_PX = 0.05
STAGNATION_MIN_SECONDS = 2.0


class MotionAccumulator:
    def __init__(self, tick_hz):
        self.minimum_run = max(1, round(STAGNATION_MIN_SECONDS * tick_hz))
        self.ticks = self.airborne = self.stationary = 0
        self.forward = self.backward = 0.0
        self.run = self.stagnation_ticks = self.stagnation_runs = 0

    def observe(self, before, after):
        dx = after.player.x - before.player.x
        self.ticks += 1
        self.airborne += int(not after.player.grounded)
        self.forward += max(0.0, dx)
        self.backward += max(0.0, -dx)
        if abs(dx) <= STATIONARY_EPSILON_PX:
            self.stationary += 1
            self.run += 1
        else:
            if self.run >= self.minimum_run:
                self.stagnation_runs += 1
                self.stagnation_ticks += self.run
            self.run = 0

    def snapshot(self):
        pending = self.run >= self.minimum_run
        return {
            "version": MOTION_VERSION, "observed_ticks": self.ticks,
            "airborne_ticks": self.airborne, "stationary_ticks": self.stationary,
            "forward_distance_px": self.forward, "backward_distance_px": self.backward,
            "stagnation_ticks": self.stagnation_ticks + (self.run if pending else 0),
            "stagnation_runs": self.stagnation_runs + int(pending),
            "stationary_epsilon_px": STATIONARY_EPSILON_PX,
            "stagnation_min_ticks": self.minimum_run,
        }


def validate_motion(value, ticks):
    if not isinstance(value, dict) or type(value.get("version")) is not int or (
        value["version"] != MOTION_VERSION
    ):
        raise ValueError("unsupported motion summary")
    for key in (
        "observed_ticks", "airborne_ticks", "stationary_ticks", "stagnation_ticks",
        "stagnation_runs", "stagnation_min_ticks",
    ):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError(f"invalid motion {key}")
    if value["observed_ticks"] != ticks or any(value[k] > ticks for k in (
        "airborne_ticks", "stationary_ticks", "stagnation_ticks", "stagnation_runs"
    )) or value["stagnation_ticks"] > value["stationary_ticks"]:
        raise ValueError("motion tick counts disagree")
    if value["stagnation_min_ticks"] < 1:
        raise ValueError("invalid stagnation interval")
    for key in ("forward_distance_px", "backward_distance_px", "stationary_epsilon_px"):
        if type(value.get(key)) not in (int, float) or not isfinite(value[key]) or value[key] < 0:
            raise ValueError(f"invalid motion {key}")
    if value["stationary_epsilon_px"] != STATIONARY_EPSILON_PX:
        raise ValueError("unknown stationary threshold")
    return value
