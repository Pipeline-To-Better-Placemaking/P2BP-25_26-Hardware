import select
import subprocess
import logging
import re

logger = logging.getLogger(__name__)

TEGRastats_RE_GPU_UTIL = re.compile(r"GR3D_FREQ\s+(\d+)%")
TEGRastats_RE_GPU_FREQ = re.compile(r"GR3D_FREQ\s+\d+%@(\d+)")
TEGRastats_RE_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")
TEGRastats_RE_GPU_TEMP = re.compile(r"gpu@([\d.]+)C")
TEGRastats_RE_CPU_TEMP = re.compile(r"cpu@([\d.]+)C")

def get_gpu_and_memory_stats():
    stats = {
        "Gpu": {
            "UtilizationPct": -1,
            "FrequencyMhz": -1,
            "TemperatureC": -1.0,
        },
        "Memory": {
            "UsedMb": -1,
            "TotalMb": -1,
        },
        "CpuTemperatureC": -1.0,
    }

    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "500"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            ready, _, _ = select.select([proc.stdout], [], [], 3.0)
            output = proc.stdout.readline().strip() if ready else ""
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

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

        # Temperatures
        gpu_temp = TEGRastats_RE_GPU_TEMP.search(output)
        if gpu_temp:
            stats["Gpu"]["TemperatureC"] = float(gpu_temp.group(1))

        cpu_temp = TEGRastats_RE_CPU_TEMP.search(output)
        if cpu_temp:
            stats["CpuTemperatureC"] = float(cpu_temp.group(1))

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
            "TemperatureC": -1.0,
        },
        "Memory": {
            "UsedMb": -1,
            "TotalMb": -1,
        },
        "CpuTemperatureC": -1.0,
    }

    gpu_mem = get_gpu_and_memory_stats()

    # merge results into one dict
    for key, val in gpu_mem.items():
        if isinstance(val, dict):
            stats[key].update(val)
        else:
            stats[key] = val

    return stats