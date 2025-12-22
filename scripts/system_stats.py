import subprocess
import logging
import re

logger = logging.getLogger(__name__)

TEGRastats_RE_GPU_UTIL = re.compile(r"GR3D_FREQ\s+(\d+)%")
TEGRastats_RE_GPU_FREQ = re.compile(r"GR3D_FREQ\s+\d+%@(\d+)")
TEGRastats_RE_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")

def get_gpu_and_memory_stats():
    stats = {
        "Gpu": {
            "UtilizationPct": -1,
            "FrequencyMhz": -1,
        },
        "Memory": {
            "UsedMb": -1,
            "TotalMb": -1,
        },
    }

    try:
        proc = subprocess.run(
            ["tegrastats", "--interval", "1000", "--count", "1"],
            capture_output=True,
            text=True,
            timeout=3,
        )

        output = proc.stdout.strip()
        if not output:
            return stats

        # GPU utilization (always present if GR3D_FREQ exists)
        gpu_util = TEGRastats_RE_GPU_UTIL.search(output)
        if gpu_util:
            stats["Gpu"]["UtilizationPct"] = int(gpu_util.group(1))

        # GPU frequency (only when GPU is active)
        gpu_freq = TEGRastats_RE_GPU_FREQ.search(output)
        if gpu_freq:
            stats["Gpu"]["FrequencyMhz"] = int(gpu_freq.group(1))

        # Memory
        ram = TEGRastats_RE_RAM.search(output)
        if ram:
            stats["Memory"]["UsedMb"] = int(ram.group(1))
            stats["Memory"]["TotalMb"] = int(ram.group(2))

        return stats

    except FileNotFoundError:
        logging.error("tegrastats not found (not a Jetson?)")
        return stats

    except Exception as e:
        logging.error(f"Failed to read GPU/memory stats: {e}")
        return stats
    
def get_system_stats():
    stats = {
        "Gpu": {
            "UtilizationPct": -1,
            "FrequencyMhz": -1,
        },
        "Memory": {
            "UsedMb": -1,
            "TotalMb": -1,
        },
    }

    gpu_mem = get_gpu_and_memory_stats()

    # merge results into one dict
    for key in gpu_mem:
        stats[key].update(gpu_mem[key])

    return stats