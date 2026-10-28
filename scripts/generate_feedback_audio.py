"""Rebuild original synthesized interaction sounds; standard library only."""

from array import array
from math import exp, pi, sin
from pathlib import Path
import random
import sys
import wave

RATE = 44100
SPECS = {
    "box": (0.16, 320, 130, 0.10),
    "brick": (0.28, 170, 55, 0.75),
    "acorn": (0.26, 760, 1450, 0.00),
    "stomp": (0.18, 180, 480, 0.06),
    "shield_break": (0.40, 1700, 240, 0.48),
    "shield_pickup": (0.38, 520, 1300, 0.00),
    "hit": (0.24, 240, 65, 0.15),
}


def main():
    directory = Path(__file__).resolve().parents[1] / "assets/audio"
    directory.mkdir(parents=True, exist_ok=True)
    for name, (duration, start, end, noise) in SPECS.items():
        rng = random.Random(name)
        samples = array("h")
        phase = 0.0
        count = round(RATE * duration)
        for i in range(count):
            t = i / RATE
            p = i / (count - 1)
            frequency = start + (end - start) * p
            phase += 2 * pi * frequency / RATE
            tone = sin(phase) * 0.75 + sin(phase * 2) * 0.25
            envelope = min(1, t / 0.006) * exp(-4 * p) * min(1, (1 - p) / 0.08)
            sample = ((1 - noise) * tone + noise * rng.uniform(-1, 1)) * envelope
            samples.append(round(sample * 22000))
        if sys.byteorder != "little":
            samples.byteswap()
        with wave.open(str(directory / f"{name}.wav"), "wb") as output:
            output.setparams((1, 2, RATE, count, "NONE", "not compressed"))
            output.writeframes(samples.tobytes())
    print(f"Generated {len(SPECS)} original WAV effects in {directory}")


if __name__ == "__main__":
    main()
