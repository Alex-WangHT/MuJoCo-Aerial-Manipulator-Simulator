# Architecture

This document summarizes the current code structure and runtime call paths.

## Runtime Call Path

The entry point is `Main.py`. It builds a drone controller, starts the MuJoCo simulation thread, and opens the ground control station.

### Class Diagram

```mermaid
classDiagram
    class Main {
        +main()
    }

    class MujocoSimulation {
        -shutdownEvent
        -droneController
        -robot
        -telemetryBuffer
        +run()
        +_launchViewer()
        +_sendSensorData()
        +_applyControl()
    }

    class Robot {
        +getSensorData()
        +applyControl()
    }

    class DroneControllerInterface {
        +setSensorData()
        +getControlInput()
        +computeControl()
    }

    class GCS {
        -telemetryBuffer
        +applyPositionTarget()
        +applyAttitudeTarget()
        +updateTelemetryPlot()
    }

    class TelemetryBuffer {
        +append()
        +snapshot()
        +clear()
    }

    Main --> MujocoSimulation
    Main --> GCS
    MujocoSimulation --> Robot
    MujocoSimulation --> DroneControllerInterface
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

    Main->>Sim: start(droneController)
    Main->>GCS: create(telemetryBuffer)

    loop each sim step
        Sim->>Sim: Robot.getSensorData()
        Sim->>Sim: droneController.setSensorData()
        Sim->>Sim: droneController.setUpdateEvent()
        Sim->>TB: append(time, position, euler)
        Sim->>Sim: droneController.getControlInput()
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

- `GCS` is created with `taskManager=None`, so it does not send control commands to the vehicle.
- The left-side target inputs in the UI only affect the plotted reference lines.
- The actual control applied to the vehicle comes from the injected `DroneControllerInterface` instance.

## Summary

The active runtime chain is:

```text
Main -> controller factory -> DroneControllerInterface
Main -> MujocoSimulation -> Robot <-> MuJoCo
Main -> GCS -> TelemetryBuffer
```
