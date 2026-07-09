# Architecture

This document summarizes the current code structure and the main runtime call paths.
It separates the system into two views:

- the current runtime path that is actually active in `Main.py`
- the full closed-loop design path that already exists in code but is not currently wired into `Main.py`

## Current Runtime Path

The current entry point is [Main.py](C:/Users/wangh/Desktop/Simulation/Main.py:1).
At the moment, it builds a multirotor controller through a controller factory, starts MuJoCo with that controller, and opens the ground control station in telemetry-only mode.

### Class Diagram

```mermaid
classDiagram
    class Main {
        +main()
    }

    class MujocoSimulation {
        -shutdownEvent
        -droneController
        -swayController
        -telemetryBuffer
        -fixedRotorThrust
        +run()
        +launchViewer()
        +_sendSensorData()
        +_applyControl()
    }

    class GCS {
        -taskManager
        -telemetryBuffer
        +applyPositionTarget()
        +applyAttitudeTarget()
        +updateTelemetryPlot()
    }

    class TelemetryBuffer {
        +append(timestamp, position, euler)
        +snapshot()
        +clear()
    }

    Main --> MujocoSimulation
    Main --> GCS
    MujocoSimulation --> TelemetryBuffer
    GCS --> TelemetryBuffer
```

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Main
    participant Sim as MujocoSimulation
    participant MJ as MuJoCo
    participant TB as TelemetryBuffer
    participant GCS

    Main->>Main: build controller via factory
    Main->>Sim: start(droneController)
    Main->>GCS: create(taskManager=None, telemetryBuffer)

    loop each sim step
        Sim->>MJ: read qpos/qvel
        MJ-->>Sim: state
        Sim->>TB: append(time, position, euler)
        Sim->>MJ: ctrl[rotor1..rotor6] = controller output
        Sim->>MJ: mj_step()
    end

    loop each UI refresh
        GCS->>TB: snapshot()
        TB-->>GCS: time, XYZ, Euler
        GCS->>GCS: draw actual curves + target dashed lines
    end
```

### Notes

- `GCS` is currently created with `taskManager=None`, so it does not send control commands to the vehicle.
- The left-side target inputs in the UI only affect the plotted reference lines in this mode.
- The actual control applied to the vehicle comes from the injected multirotor controller instance.

## Legacy Closed-Loop Design Path

The project still contains legacy position and attitude controller modules under `src/drone/droneControllers/`.
They are not part of the current `Main.py` runtime path, which now uses only `DroneControllerInterface`.

### Class Diagram

```mermaid
classDiagram
    class GCS {
        -taskManager
        -telemetryBuffer
        +applyPositionTarget()
        +applyAttitudeTarget()
        +applyAllTargets()
    }

    class TaskManager {
        -gcsMessageQueue
        -droneControlQueue
        -droneControlMessage
        +enqueue(command)
        +executeCommand(message)
        +setPoint(...)
    }

    class CoordinatesController {
        +setCoordinatesData(data)
        +getAttitudeReference()
        +getRotorForce()
    }

    class DroneStabilityController {
        +setStabilityData(data)
        +getRotorAdjustments()
    }

    class MujocoSimulation {
        +_sendSensorData()
        +_applyControl()
    }

    class Command
    class SensorData

    GCS --> TaskManager
    TaskManager --> Command
    CoordinatesController --> SensorData
    DroneStabilityController --> SensorData
    MujocoSimulation --> CoordinatesController
    MujocoSimulation --> DroneStabilityController
```

### Sequence Diagram

```mermaid
sequenceDiagram
    participant GCS
    participant TM as TaskManager
    participant CC as CoordinatesController
    participant SC as DroneStabilityController
    participant Sim as MujocoSimulation
    participant MJ as MuJoCo

    GCS->>TM: enqueue(Command:setpoint)
    TM->>TM: Command -> target state

    loop each sim step
        Sim->>MJ: read qpos/qvel
        MJ-->>Sim: state
        Sim->>CC: setCoordinatesData(position, velocity, positionReference, yawReference)
        CC->>CC: position loop + velocity loop
        CC-->>Sim: attitudeReference, rotorForce

        Sim->>SC: setStabilityData(orientation, orientationReference, angularVelocity)
        SC->>SC: angle loop + rate loop
        SC-->>Sim: rotorAdjustments

        Sim->>MJ: write ctrl[]
        Sim->>MJ: mj_step()
    end
```

### Legacy Control Composition

In the legacy cascaded design, the nested controllers are combined as:

```text
u_i = base_thrust + rotor_adjustments[i]
```

where:

- `base_thrust` comes from `CoordinatesController.getRotorForce()`
- `rotor_adjustments` comes from `DroneStabilityController.getRotorAdjustments()`

This means:

- the outer position loop generates a collective thrust level and an attitude reference
- the inner attitude loop converts the attitude tracking error into per-rotor differential inputs

## Message and Data Flow

The main data objects live in [include/Messages.py](C:/Users/wangh/Desktop/Simulation/include/Messages.py:1):

- `Command`: high-level UI command from `GCS` to `TaskManager`
- `ControlMessage`: legacy target state previously used by the old cascaded controller path
- `SensorData`: vehicle feedback sent from `MujocoSimulation` to the active controller
- `ControlAction`: legacy actuator-command container used by the old cascaded controller path

You can think of them as four layers:

```text
UI command layer:     Command
control target layer: ControlMessage
feedback layer:       SensorData
actuator layer:       ControlAction
```

## Summary

The currently active runtime chain is:

```text
Main -> controller factory -> DroneControllerInterface
Main -> MujocoSimulation -> MuJoCo
Main -> GCS -> TelemetryBuffer
```

The legacy cascaded closed-loop chain is:

```text
GCS -> TaskManager
TaskManager -> CoordinatesController / DroneStabilityController
CoordinatesController / DroneStabilityController -> MujocoSimulation -> MuJoCo
MuJoCo -> MujocoSimulation -> SensorData -> controller modules
```
