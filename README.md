# Cable Suspended Dual Arm Aerial Manipulator Framework

This repository is a code framework based on Annex B of the referenced thesis.
It preserves the annex module boundaries and thread-level interfaces.
`Main.py` currently wires MuJoCo to a `DroneControllerInterface` controller. The default controller is an open-loop fixed-thrust multirotor controller.

- `Main.py`: accepts a multirotor controller factory and starts MuJoCo with the resulting controller instance.
- `include/Messages.py`: shared command, sensor, and actuator messages.
- `include/PIDController.py`: vector PID used by the cascaded controllers.
- `src/TaskManager.py`: translates GCS commands into `ControlMessage` objects.
- `src/GroundControlStation.py`: Tkinter telemetry window with position/attitude targets and live actual-vs-target plots.
- `src/MujocoSimulation.py`: simulation loop and controller/simulator data bridge.
- `src/drone/DroneControllerInterface.py`: thread-based abstract drone-controller interface for feedback, target pose, and control input `u`.
- `src/drone/droneControllers/`: 3D position/collective-thrust and attitude controllers.
- `src/manipulator/SwayControl.py`: joint-space cascaded controller and Annex B sway cases.
- `cython_modules/fast_mujoco_utils.py`: Python fallback for the Cython extraction helper described in the annex.
- `models/`: placeholder MJCF files with the same names as the annex final models.

Run the current fixed-thrust test:

```powershell
python Main.py
```

The XML files in `models/` are intentionally lightweight placeholders. Replace
them with the complete Annex B.1 MJCF models and mesh assets when those files
are available.
