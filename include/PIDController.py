from __future__ import annotations

import numpy as np


class PIDController:
    def __init__(self, kp, ki=0.0, kd=0.0, outputLimit=None, integralLimit=None):
        self.kp = np.asarray(kp, dtype=float)
        self.ki = np.asarray(ki, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.output_limit = outputLimit
        self.integral_limit = integralLimit
        self._integral = np.zeros_like(self.kp, dtype=float)
        self._previous_error = np.zeros_like(self.kp, dtype=float)
        self._has_previous = False

    def reset(self) -> None:
        self._integral = np.zeros_like(self._integral, dtype=float)
        self._previous_error = np.zeros_like(self._previous_error, dtype=float)
        self._has_previous = False

    def compute(self, error, dt: float):
        error = np.asarray(error, dtype=float)
        dt = max(float(dt), 1e-9)

        candidate_integral = self._integral + error * dt
        if self.integral_limit is not None:
            limit = np.asarray(self.integral_limit, dtype=float)
            candidate_integral = np.clip(candidate_integral, -limit, limit)

        derivative = np.zeros_like(error)
        if self._has_previous:
            derivative = (error - self._previous_error) / dt
        self._previous_error = error.copy()
        self._has_previous = True

        output = self.kp * error + self.ki * candidate_integral + self.kd * derivative
        if self.output_limit is not None:
            limit = np.asarray(self.output_limit, dtype=float)
            pushing_upper = (output > limit) & (error > 0.0)
            pushing_lower = (output < -limit) & (error < 0.0)
            windup_mask = pushing_upper | pushing_lower
            candidate_integral = np.where(windup_mask, self._integral, candidate_integral)
            output = self.kp * error + self.ki * candidate_integral + self.kd * derivative
            output = np.clip(output, -limit, limit)
        self._integral = candidate_integral
        return output
