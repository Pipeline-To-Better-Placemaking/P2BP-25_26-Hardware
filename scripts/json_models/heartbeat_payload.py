from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import time

# #  return {
#         "Id": "0",
#         "ProjectId": "0",
#         "DeviceId": os.uname().nodename,
#         "Timestamp": int(time.time()),
#         "Services": get_all_service_states(),
#         "System": get_system_stats(),
#     }

@dataclass
class ServiceState:
    Active: str = "unknown"
    Sub: str = "unknown"

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
    Id: str
    ProjectId: str
    DeviceId: str
    Timestamp: int
    Services: Dict[str, ServiceState]
    System: SystemStats

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def build(
        project_id: str, 
        device_id: str, 
        services: Dict[str, ServiceState], 
        system: SystemStats
    ) -> "HeartbeatPayload":
        return HeartbeatPayload(
            Id="0",
            ProjectId=project_id,
            DeviceId=device_id,
            Timestamp=int(time.time()),
            Services=services,
            System=system,
        )