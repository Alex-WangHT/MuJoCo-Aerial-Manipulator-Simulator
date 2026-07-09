from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from include.Messages import Command
from include.Telemetry import TelemetryBuffer


class GCS:
    def __init__(
        self,
        root: tk.Tk,
        taskManager=None,
        shutdownEvent=None,
        telemetryBuffer: TelemetryBuffer | None = None,
    ):
        self._root = root
        self._taskManager = taskManager
        self._shutdownEvent = shutdownEvent
        self._telemetryBuffer = telemetryBuffer or TelemetryBuffer()
        self._positionTarget = np.array([0.0, 0.0, 1.5], dtype=float)
        self._attitudeTargetDeg = np.zeros(3)
        self._root.title("Ground Control Station")
        self._root.minsize(1120, 620)
        self._buildInterface()
        self._root.protocol("WM_DELETE_WINDOW", self.stop)
        self.checkShutdown()

    def _buildInterface(self) -> None:
        self._root.columnconfigure(0, weight=0)
        self._root.columnconfigure(1, weight=1)
        self._root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self._root, padding=12)
        controls.grid(row=0, column=0, sticky="nsw")

        plot_frame = ttk.Frame(self._root, padding=(0, 12, 12, 12))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self._buildTargetPanel(controls)
        self._buildTelemetryPanel(plot_frame)

    def _buildTargetPanel(self, frame: ttk.Frame) -> None:
        for col in range(2):
            frame.columnconfigure(col, weight=1)

        self.xVar = tk.StringVar(value=f"{self._positionTarget[0]:.2f}")
        self.yVar = tk.StringVar(value=f"{self._positionTarget[1]:.2f}")
        self.zVar = tk.StringVar(value=f"{self._positionTarget[2]:.2f}")
        self.rollVar = tk.StringVar(value="0.0")
        self.pitchVar = tk.StringVar(value="0.0")
        self.yawVar = tk.StringVar(value="0.0")
        mode_text = "control enabled" if self._taskManager is not None else "plot target only"
        self.targetStatusVar = tk.StringVar(value=mode_text)

        ttk.Label(frame, text="Position target").grid(row=0, column=0, columnspan=2, sticky="w")
        self._addField(frame, 1, "X [m]", self.xVar)
        self._addField(frame, 2, "Y [m]", self.yVar)
        self._addField(frame, 3, "Z [m]", self.zVar)
        ttk.Button(frame, text="Apply position", command=self.applyPositionTarget).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(6, 14)
        )

        ttk.Label(frame, text="Attitude target").grid(row=5, column=0, columnspan=2, sticky="w")
        self._addField(frame, 6, "Roll [deg]", self.rollVar)
        self._addField(frame, 7, "Pitch [deg]", self.pitchVar)
        self._addField(frame, 8, "Yaw [deg]", self.yawVar)
        ttk.Button(frame, text="Apply attitude", command=self.applyAttitudeTarget).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=(6, 8)
        )
        ttk.Button(frame, text="Apply all targets", command=self.applyAllTargets).grid(
            row=10, column=0, columnspan=2, sticky="ew"
        )

        ttk.Separator(frame).grid(row=11, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Button(frame, text="Clear telemetry", command=self.clearTelemetryPlot).grid(
            row=12, column=0, columnspan=2, sticky="ew"
        )
        ttk.Label(frame, textvariable=self.targetStatusVar, wraplength=220).grid(
            row=13, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

    @staticmethod
    def _addField(frame: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=variable, width=10).grid(row=row, column=1, sticky="ew", pady=2)

    def _buildTelemetryPanel(self, frame: ttk.Frame) -> None:
        self.telemetryStatusVar = tk.StringVar(value="Waiting for simulation data")
        ttk.Label(frame, textvariable=self.telemetryStatusVar).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self._telemetryFigure = Figure(figsize=(8.5, 5.4), dpi=100)
        self._positionAxes = self._telemetryFigure.add_subplot(211)
        self._eulerAxes = self._telemetryFigure.add_subplot(212, sharex=self._positionAxes)
        self._telemetryCanvas = FigureCanvasTkAgg(self._telemetryFigure, master=frame)
        telemetry_widget = self._telemetryCanvas.get_tk_widget()
        telemetry_widget.configure(width=860, height=520)
        telemetry_widget.grid(row=0, column=0, sticky="nsew")

        self._positionLines = []
        self._positionTargetLines = []
        self._eulerLines = []
        self._eulerTargetLines = []
        self._setupTelemetryAxes()
        self.updateTelemetryPlot()

    def _setupTelemetryAxes(self) -> None:
        self._positionAxes.clear()
        self._eulerAxes.clear()
        colors = ["#1f77b4", "#2ca02c", "#d62728"]
        position_labels = ["X", "Y", "Z"]
        euler_labels = ["Roll", "Pitch", "Yaw"]
        self._positionLines = [
            self._positionAxes.plot([], [], label=label, color=color)[0]
            for label, color in zip(position_labels, colors)
        ]
        self._positionTargetLines = [
            self._positionAxes.plot([], [], "--", label=f"{label} target", color=color, alpha=0.55)[0]
            for label, color in zip(position_labels, colors)
        ]
        self._eulerLines = [
            self._eulerAxes.plot([], [], label=label, color=color)[0]
            for label, color in zip(euler_labels, colors)
        ]
        self._eulerTargetLines = [
            self._eulerAxes.plot([], [], "--", label=f"{label} target", color=color, alpha=0.55)[0]
            for label, color in zip(euler_labels, colors)
        ]
        self._positionAxes.set_title("Position actual vs target")
        self._positionAxes.set_ylabel("Position [m]")
        self._positionAxes.grid(True)
        self._positionAxes.legend(loc="upper right", ncol=3, fontsize=8)
        self._eulerAxes.set_title("Euler angles actual vs target")
        self._eulerAxes.set_xlabel("Time [s]")
        self._eulerAxes.set_ylabel("Angle [deg]")
        self._eulerAxes.grid(True)
        self._eulerAxes.legend(loc="upper right", ncol=3, fontsize=8)
        self._telemetryFigure.tight_layout()

    def applyPositionTarget(self) -> None:
        try:
            self._positionTarget = np.array(
                [float(self.xVar.get()), float(self.yVar.get()), float(self.zVar.get())],
                dtype=float,
            )
            self._sendSetpoint(
                coordinates=tuple(float(v) for v in self._positionTarget),
                attitude=None,
                attitudeOverride=False,
            )
            self.targetStatusVar.set(self._targetSummary("position target applied"))
        except ValueError as exc:
            self.targetStatusVar.set(f"Invalid position target: {exc}")

    def applyAttitudeTarget(self) -> None:
        try:
            self._attitudeTargetDeg = np.array(
                [float(self.rollVar.get()), float(self.pitchVar.get()), float(self.yawVar.get())],
                dtype=float,
            )
            attitude_rad = tuple(float(v) for v in np.deg2rad(self._attitudeTargetDeg))
            self._sendSetpoint(coordinates=None, attitude=attitude_rad, attitudeOverride=True)
            self.targetStatusVar.set(self._targetSummary("attitude target applied"))
        except ValueError as exc:
            self.targetStatusVar.set(f"Invalid attitude target: {exc}")

    def applyAllTargets(self) -> None:
        try:
            self._positionTarget = np.array(
                [float(self.xVar.get()), float(self.yVar.get()), float(self.zVar.get())],
                dtype=float,
            )
            self._attitudeTargetDeg = np.array(
                [float(self.rollVar.get()), float(self.pitchVar.get()), float(self.yawVar.get())],
                dtype=float,
            )
            attitude_rad = tuple(float(v) for v in np.deg2rad(self._attitudeTargetDeg))
            self._sendSetpoint(
                tuple(float(v) for v in self._positionTarget),
                attitude_rad,
                attitudeOverride=True,
            )
            self.targetStatusVar.set(self._targetSummary("all targets applied"))
        except ValueError as exc:
            self.targetStatusVar.set(f"Invalid target: {exc}")

    def _sendSetpoint(
        self,
        coordinates: tuple[float, float, float] | None,
        attitude: tuple[float, float, float] | None,
        attitudeOverride: bool | None,
    ) -> None:
        if self._taskManager is None:
            return
        self._taskManager.enqueue(
            Command(
                "setpoint",
                coordinates=coordinates,
                attitude=attitude,
                attitudeOverride=attitudeOverride,
            )
        )

    def _targetSummary(self, prefix: str) -> str:
        mode_text = "sent to controller" if self._taskManager is not None else "plot target only"
        return (
            f"{prefix} ({mode_text})\n"
            f"XYZ {self._positionTarget[0]:.2f}, {self._positionTarget[1]:.2f}, {self._positionTarget[2]:.2f} m\n"
            f"RPY {self._attitudeTargetDeg[0]:.1f}, {self._attitudeTargetDeg[1]:.1f}, "
            f"{self._attitudeTargetDeg[2]:.1f} deg"
        )

    def clearTelemetryPlot(self) -> None:
        self._telemetryBuffer.clear()
        self._setupTelemetryAxes()
        self._telemetryCanvas.draw_idle()
        self.telemetryStatusVar.set("Telemetry cleared")

    def updateTelemetryPlot(self) -> None:
        if self._shutdownEvent is not None and self._shutdownEvent.is_set():
            return
        time, position, euler = self._telemetryBuffer.snapshot()
        if len(time) > 0:
            relative_time = time - time[0]
            euler_deg = np.rad2deg(euler)
            for i, line in enumerate(self._positionLines):
                line.set_data(relative_time, position[:, i])
            for i, line in enumerate(self._positionTargetLines):
                line.set_data(relative_time, np.full_like(relative_time, self._positionTarget[i]))
            for i, line in enumerate(self._eulerLines):
                line.set_data(relative_time, euler_deg[:, i])
            for i, line in enumerate(self._eulerTargetLines):
                line.set_data(relative_time, np.full_like(relative_time, self._attitudeTargetDeg[i]))

            self._positionAxes.relim()
            self._positionAxes.autoscale_view()
            self._eulerAxes.relim()
            self._eulerAxes.autoscale_view()
            self._telemetryCanvas.draw_idle()
            latest_position = position[-1]
            latest_euler = euler_deg[-1]
            self.telemetryStatusVar.set(
                "XYZ: "
                f"{latest_position[0]:.2f}, {latest_position[1]:.2f}, {latest_position[2]:.2f} m | "
                "RPY: "
                f"{latest_euler[0]:.1f}, {latest_euler[1]:.1f}, {latest_euler[2]:.1f} deg"
            )
        else:
            self.telemetryStatusVar.set("Waiting for simulation data")
        self._root.after(100, self.updateTelemetryPlot)

    def checkShutdown(self) -> None:
        if self._shutdownEvent is not None and self._shutdownEvent.is_set():
            self._root.quit()
            return
        self._root.after(100, self.checkShutdown)

    def stop(self) -> None:
        if self._shutdownEvent is not None:
            self._shutdownEvent.set()
        self._root.quit()
