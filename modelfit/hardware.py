"""Hardware detection across NVIDIA / Apple / AMD / generic CPU systems.

Outputs a normalized HardwareProfile that the scoring engine consumes.
Memory bandwidth numbers come from a small lookup table of common chips
(we use the matched model when available, otherwise a sane platform default).
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import psutil

GB = 1024**3
MB = 1024**2


# ---------------------------------------------------------------------------
# Memory-bandwidth lookups (GB/s). Numbers are advertised peak; real-world is
# typically 60-80% but we apply a utilization factor in the speed estimator.
# ---------------------------------------------------------------------------
GPU_BANDWIDTH = {
    # NVIDIA consumer
    "rtx 5090": 1792, "rtx 5080": 960, "rtx 5070 ti": 896, "rtx 5070": 672,
    "rtx 4090": 1008, "rtx 4080 super": 736, "rtx 4080": 717, "rtx 4070 ti super": 672,
    "rtx 4070 ti": 504, "rtx 4070 super": 504, "rtx 4070": 504, "rtx 4060 ti": 288,
    "rtx 4060": 272,
    "rtx 3090 ti": 1008, "rtx 3090": 936, "rtx 3080 ti": 912, "rtx 3080": 760,
    "rtx 3070 ti": 608, "rtx 3070": 448, "rtx 3060 ti": 448, "rtx 3060": 360,
    "rtx 2080 ti": 616, "rtx 2080 super": 496, "rtx 2080": 448, "rtx 2070 super": 448,
    "rtx 2070": 448, "rtx 2060 super": 448, "rtx 2060": 336,
    # NVIDIA data-center
    "h200": 4800, "h100": 3350, "a100": 2039, "l40s": 864, "l40": 864,
    "a40": 696, "a10": 600, "a6000": 768, "a5000": 768, "a4000": 448,
    "v100": 900, "t4": 320,
    # AMD
    "rx 7900 xtx": 960, "rx 7900 xt": 800, "rx 7800 xt": 624,
    "rx 6900 xt": 512, "rx 6800 xt": 512, "rx 6700 xt": 384,
    "mi300x": 5300, "mi250x": 3276, "mi210": 1638,
    # Apple unified
    "m1": 68, "m1 pro": 200, "m1 max": 400, "m1 ultra": 800,
    "m2": 100, "m2 pro": 200, "m2 max": 400, "m2 ultra": 800,
    "m3": 100, "m3 pro": 150, "m3 max": 400,
    "m4": 120, "m4 pro": 273, "m4 max": 546,
    # Intel discrete
    "arc a770": 560, "arc a750": 512, "arc b580": 456,
}

# CPU memory bandwidth defaults by platform/generation (GB/s).
CPU_BANDWIDTH_DEFAULTS = {
    "ddr5": 80.0,
    "ddr4": 50.0,
    "ddr3": 25.0,
    "apple_silicon": 100.0,
    "unknown": 40.0,
}


@dataclass
class GPU:
    name: str
    vendor: str  # nvidia | apple | amd | intel | unknown
    vram_bytes: int
    bandwidth_gbps: float
    compute_capability: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vram_gb"] = round(self.vram_bytes / GB, 2)
        return d


@dataclass
class HardwareProfile:
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    cpu_freq_mhz: float
    ram_bytes: int
    ram_available_bytes: int
    ram_bandwidth_gbps: float
    gpus: List[GPU] = field(default_factory=list)
    platform: str = ""
    arch: str = ""
    unified_memory: bool = False  # Apple Silicon: RAM is VRAM
    notes: List[str] = field(default_factory=list)

    @property
    def total_vram_bytes(self) -> int:
        if self.unified_memory and self.gpus:
            # On Apple Silicon, ~75% of unified RAM is usable for the GPU.
            return int(self.ram_bytes * 0.75)
        return sum(g.vram_bytes for g in self.gpus)

    @property
    def best_gpu(self) -> Optional[GPU]:
        return max(self.gpus, key=lambda g: g.vram_bytes) if self.gpus else None

    @property
    def effective_bandwidth_gbps(self) -> float:
        """Bandwidth of the fastest memory tier we'd actually use."""
        if self.gpus:
            return max(g.bandwidth_gbps for g in self.gpus)
        return self.ram_bandwidth_gbps

    def to_dict(self) -> dict:
        return {
            "cpu_name": self.cpu_name,
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "ram_gb": round(self.ram_bytes / GB, 2),
            "ram_available_gb": round(self.ram_available_bytes / GB, 2),
            "ram_bandwidth_gbps": self.ram_bandwidth_gbps,
            "platform": self.platform,
            "arch": self.arch,
            "unified_memory": self.unified_memory,
            "total_vram_gb": round(self.total_vram_bytes / GB, 2),
            "effective_bandwidth_gbps": self.effective_bandwidth_gbps,
            "gpus": [g.to_dict() for g in self.gpus],
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# CPU detection
# ---------------------------------------------------------------------------
def _detect_cpu_name() -> str:
    sysname = platform.system()
    try:
        if sysname == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2
            ).strip()
            if out:
                return out
        elif sysname == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass
        elif sysname == "Windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], text=True, timeout=3
            )
            lines = [l.strip() for l in out.splitlines() if l.strip() and "Name" not in l]
            if lines:
                return lines[0]
    except Exception:
        pass
    return platform.processor() or platform.machine() or "Unknown CPU"


def _guess_cpu_bandwidth(cpu_name: str) -> float:
    """Heuristic from CPU model string."""
    n = cpu_name.lower()
    # Apple Silicon → unified memory; treated separately, but provide a default.
    if "apple" in n and ("m1" in n or "m2" in n or "m3" in n or "m4" in n):
        return CPU_BANDWIDTH_DEFAULTS["apple_silicon"]
    # DDR5 on recent platforms
    ddr5_markers = [
        "13th gen", "14th gen", "15th gen", "core ultra",
        "i3-13", "i5-13", "i7-13", "i9-13",
        "i3-14", "i5-14", "i7-14", "i9-14",
        "ryzen 7000", "ryzen 8000", "ryzen 9000", "ryzen ai",
        "7950x", "7900x", "7800x", "7700x", "7600x",
        "9950x", "9900x", "9700x", "9600x",
        "epyc 9", "xeon w-3", "xeon platinum 8",
    ]
    if any(m in n for m in ddr5_markers):
        return CPU_BANDWIDTH_DEFAULTS["ddr5"]
    return CPU_BANDWIDTH_DEFAULTS["ddr4"]


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def _match_gpu_bandwidth(name: str) -> float:
    n = name.lower()
    # Strip "NVIDIA GeForce", "AMD Radeon", etc. for matching.
    n_clean = re.sub(r"\b(nvidia|geforce|amd|radeon|intel|arc|apple)\b", "", n).strip()
    # Longest-key-first match so "rtx 4080 super" beats "rtx 4080".
    keys = sorted(GPU_BANDWIDTH.keys(), key=len, reverse=True)
    for k in keys:
        if k in n_clean or k in n:
            return GPU_BANDWIDTH[k]
    # Family defaults.
    if "h100" in n: return 3350
    if "a100" in n: return 2039
    if "rtx 40" in n: return 600
    if "rtx 30" in n: return 500
    if "rtx 20" in n: return 400
    if "rtx" in n: return 350
    if "rx 7" in n: return 600
    if "rx 6" in n: return 450
    if "apple" in n: return 200
    return 250  # generic guess


def _detect_nvidia_gpus() -> List[GPU]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, timeout=4,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            vram_mb = float(parts[1])
        except ValueError:
            continue
        cc = parts[2] if len(parts) > 2 else None
        gpus.append(GPU(
            name=name,
            vendor="nvidia",
            vram_bytes=int(vram_mb * MB),
            bandwidth_gbps=_match_gpu_bandwidth(name),
            compute_capability=cc,
        ))
    return gpus


def _detect_amd_gpus() -> List[GPU]:
    if not shutil.which("rocm-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"],
            text=True, timeout=4,
        )
        data = json.loads(out)
    except Exception:
        return []
    gpus = []
    for key, info in data.items():
        if not key.startswith("card") and not key.startswith("GPU"):
            continue
        name = info.get("Card series") or info.get("Card model") or info.get("Device Name") or "AMD GPU"
        vram_str = info.get("VRAM Total Memory (B)") or info.get("vram_total") or "0"
        try:
            vram_bytes = int(vram_str)
        except (ValueError, TypeError):
            vram_bytes = 0
        if vram_bytes == 0:
            continue
        gpus.append(GPU(
            name=name, vendor="amd",
            vram_bytes=vram_bytes,
            bandwidth_gbps=_match_gpu_bandwidth(name),
        ))
    return gpus


def _detect_apple_gpu(ram_bytes: int, cpu_name: str) -> List[GPU]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    # Pull GPU/chip name out of system_profiler.
    chip = cpu_name  # Apple sets machdep.cpu.brand_string to e.g. "Apple M2 Pro".
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"], text=True, timeout=4
        )
        data = json.loads(out)
        items = data.get("SPDisplaysDataType") or []
        if items:
            chip = items[0].get("sppci_model") or chip
    except Exception:
        pass
    return [GPU(
        name=chip,
        vendor="apple",
        vram_bytes=int(ram_bytes * 0.75),
        bandwidth_gbps=_match_gpu_bandwidth(chip),
    )]


def _detect_intel_gpu() -> List[GPU]:
    # Best-effort: only flag if Arc is present via lspci on Linux.
    if platform.system() != "Linux":
        return []
    if not shutil.which("lspci"):
        return []
    try:
        out = subprocess.check_output(["lspci"], text=True, timeout=3)
    except Exception:
        return []
    gpus = []
    for line in out.splitlines():
        if "Intel" in line and ("Arc" in line or "Xe" in line):
            name = line.split(":", 2)[-1].strip()
            gpus.append(GPU(
                name=name, vendor="intel",
                vram_bytes=8 * GB,  # rough; lspci doesn't tell us
                bandwidth_gbps=_match_gpu_bandwidth(name),
            ))
    return gpus


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def detect_hardware() -> HardwareProfile:
    cpu_name = _detect_cpu_name()
    ram = psutil.virtual_memory()
    freq = psutil.cpu_freq()

    profile = HardwareProfile(
        cpu_name=cpu_name,
        cpu_cores=psutil.cpu_count(logical=False) or 1,
        cpu_threads=psutil.cpu_count(logical=True) or 1,
        cpu_freq_mhz=float(freq.max if freq and freq.max else (freq.current if freq else 0.0)),
        ram_bytes=ram.total,
        ram_available_bytes=ram.available,
        ram_bandwidth_gbps=_guess_cpu_bandwidth(cpu_name),
        platform=platform.system(),
        arch=platform.machine(),
        unified_memory=(platform.system() == "Darwin" and platform.machine() == "arm64"),
    )

    gpus: List[GPU] = []
    gpus.extend(_detect_nvidia_gpus())
    gpus.extend(_detect_amd_gpus())
    gpus.extend(_detect_apple_gpu(profile.ram_bytes, cpu_name))
    gpus.extend(_detect_intel_gpu())
    profile.gpus = gpus

    if not gpus:
        profile.notes.append("No GPU detected — falling back to CPU inference.")
    if profile.unified_memory:
        profile.notes.append("Apple Silicon detected — unified memory shared between CPU and GPU.")
    return profile


def hardware_from_dict(d: dict) -> HardwareProfile:
    """Reconstruct a profile from a dict (for the REST /rank endpoint)."""
    ram_gb = d.get("ram_gb", 16)
    ram_bytes = int(float(ram_gb) * GB)
    gpus_in = d.get("gpus") or []
    gpus = []
    for g in gpus_in:
        name = g.get("name", "Unknown GPU")
        vram_gb = float(g.get("vram_gb", 0))
        bw = float(g.get("bandwidth_gbps") or _match_gpu_bandwidth(name))
        gpus.append(GPU(
            name=name,
            vendor=g.get("vendor", "unknown"),
            vram_bytes=int(vram_gb * GB),
            bandwidth_gbps=bw,
        ))
    return HardwareProfile(
        cpu_name=d.get("cpu_name", "Custom CPU"),
        cpu_cores=int(d.get("cpu_cores", 8)),
        cpu_threads=int(d.get("cpu_threads", 16)),
        cpu_freq_mhz=float(d.get("cpu_freq_mhz", 3000.0)),
        ram_bytes=ram_bytes,
        ram_available_bytes=int(float(d.get("ram_available_gb", ram_gb)) * GB),
        ram_bandwidth_gbps=float(d.get("ram_bandwidth_gbps", 50.0)),
        gpus=gpus,
        platform=d.get("platform", "custom"),
        arch=d.get("arch", "x86_64"),
        unified_memory=bool(d.get("unified_memory", False)),
    )
