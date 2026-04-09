from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
import time

@dataclass
class ServiceState:
    Active: str = "unknown"
    Sub: str = "unknown"

@dataclass
class CameraState:
    Mac: str = ""
    Ip: str = ""
    Resolution: List[int] = field(default_factory=lambda: [0, 0])
    Enabled: bool = False

@dataclass
class GpuStats:
    UtilizationPct: int = -1
    FrequencyMhz: int = -1
    TemperatureC: float = -1.0

@dataclass
class MemoryStats:
    UsedMb: int = -1
    TotalMb: int = -1

@dataclass
class SystemStats:
    Gpu: GpuStats = field(default_factory=GpuStats)
    Memory: MemoryStats = field(default_factory=MemoryStats)
    Disk: List[Dict[str, Any]] = field(default_factory=list)
    CpuTemperatureC: float = -1.0


@dataclass
class LidarHealth:
    SensorId: str = ""
    Connected: bool = False
    DevicePath: str = ""
    UsbHint: str = ""
    LastError: str = ""


@dataclass
class PiCompanionHealth:
    Configured: bool = False
    Host: str = ""
    Reachable: bool = False
    LatencyMs: int = -1
    LastError: str = ""


@dataclass
class HeartbeatPayload:
    Timestamp: int
    Services: Dict[str, ServiceState]
    Cameras: Dict[str, CameraState]
    System: SystemStats
    IntrinsicsCalibration: Dict[str, Any] = field(default_factory=dict)
    Lidars: Dict[str, LidarHealth] = field(default_factory=dict)
    PiCompanion: PiCompanionHealth = field(default_factory=PiCompanionHealth)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def build(
        services: Dict[str, ServiceState],
        cameras: Dict[str, CameraState],
        system: SystemStats,
        intrinsics_calibration: Dict[str, Any] | None = None,
        lidars: Dict[str, LidarHealth] | None = None,
        pi_companion: PiCompanionHealth | None = None,
    ) -> "HeartbeatPayload":
        return HeartbeatPayload(
            Timestamp=int(time.time()),
            Services=services,
            Cameras=cameras,
            System=system,
            IntrinsicsCalibration=intrinsics_calibration or {},
            Lidars=lidars or {},
            PiCompanion=pi_companion or PiCompanionHealth(),
        )