from dataclasses import dataclass, field, asdict
from typing import Dict, List
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
#    temperature: int

@dataclass
class MemoryStats:
    UsedMb: int = -1
    TotalMb: int = -1

@dataclass
class SystemStats:
#    CpuPct: float
    Gpu: GpuStats = field(default_factory=GpuStats)
    Memory: MemoryStats = field(default_factory=MemoryStats)


@dataclass
class HeartbeatPayload:
    Timestamp: int
    Services: Dict[str, ServiceState]
    Cameras: Dict[str, CameraState]
    System: SystemStats

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def build(
        services: Dict[str, ServiceState],
        cameras: Dict[str, CameraState],
        system: SystemStats,
    ) -> "HeartbeatPayload":
        return HeartbeatPayload(
            Timestamp=int(time.time()),
            Services=services,
            Cameras=cameras,
            System=system,
        )