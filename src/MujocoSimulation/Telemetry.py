from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class TelemetrySample:
    timestamp: float
    position: np.ndarray
    euler: np.ndarray


class TelemetryBuffer:
    def __init__(self, maxSamples: int = 5000):
        self._maxSamples = maxSamples
        self._samples: list[TelemetrySample] = []
        self._lock = threading.Lock()

    def append(self, timestamp: float, position, euler) -> None:
        sample = TelemetrySample(
            timestamp=float(timestamp),
            position=np.asarray(position, dtype=float).copy(),
            euler=np.asarray(euler, dtype=float).copy(),
        )
        with self._lock:
            self._samples.append(sample)
            if len(self._samples) > self._maxSamples:
                del self._samples[: len(self._samples) - self._maxSamples]

    def snapshot(self):
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return np.array([]), np.empty((0, 3)), np.empty((0, 3))
        time = np.array([sample.timestamp for sample in samples], dtype=float)
        position = np.vstack([sample.position for sample in samples])
        euler = np.vstack([sample.euler for sample in samples])
        return time, position, euler

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()
