// macOS hardware detection: sysctlbyname for CPU/RAM/freq, IOKit Metal API
// for GPU. Used when MODELFIT_OS_MACOS is defined.
#if defined(__APPLE__)

#include "modelfit/hardware.h"

#include <array>
#include <cstring>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/utsname.h>

namespace modelfit {

static std::string sysctl_string(const char* name) {
    char buf[512] = {0};
    size_t size = sizeof(buf);
    if (sysctlbyname(name, &buf, &size, nullptr, 0) == 0) {
        return std::string(buf);
    }
    return "";
}

static std::int64_t sysctl_int64(const char* name) {
    std::int64_t value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) == 0) {
        return value;
    }
    return 0;
}

static int sysctl_int(const char* name) {
    int value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) == 0) {
        return value;
    }
    return 0;
}

HardwareProfile detect_hardware() {
    HardwareProfile hw;
    hw.platform = "Darwin";

    // Architecture
    struct utsname uts {};
    if (uname(&uts) == 0) {
        hw.arch = uts.machine;
    }
    hw.unified_memory = (hw.arch == "arm64");

    // CPU
    hw.cpu_name = sysctl_string("machdep.cpu.brand_string");
    if (hw.cpu_name.empty()) hw.cpu_name = "Apple CPU";
    hw.cpu_cores = sysctl_int("hw.physicalcpu");
    hw.cpu_threads = sysctl_int("hw.logicalcpu");
    if (hw.cpu_cores == 0) hw.cpu_cores = sysctl_int("hw.ncpu");
    if (hw.cpu_threads == 0) hw.cpu_threads = hw.cpu_cores;

    // CPU frequency — Apple Silicon doesn't expose hw.cpufrequency_max in
    // the public sysctl namespace; fall back to a known per-chip ballpark.
    auto freq_int = sysctl_int64("hw.cpufrequency_max");
    if (freq_int == 0) freq_int = sysctl_int64("hw.cpufrequency");
    if (freq_int == 0) {
        // Per-chip defaults; rough but better than 0.
        if (hw.cpu_name.find("M4") != std::string::npos) hw.cpu_freq_mhz = 4400;
        else if (hw.cpu_name.find("M3") != std::string::npos) hw.cpu_freq_mhz = 4050;
        else if (hw.cpu_name.find("M2") != std::string::npos) hw.cpu_freq_mhz = 3500;
        else if (hw.cpu_name.find("M1") != std::string::npos) hw.cpu_freq_mhz = 3200;
        else hw.cpu_freq_mhz = 3000;
    } else {
        hw.cpu_freq_mhz = static_cast<double>(freq_int) / 1e6;
    }

    // Memory
    hw.ram_bytes = sysctl_int64("hw.memsize");
    // No direct "available memory" API on macOS without vm_statistics —
    // approximate as the system memory we just probed (good enough for sizing).
    hw.ram_available_bytes = hw.ram_bytes;

    // RAM bandwidth — Apple Silicon unified memory has chip-specific bandwidth.
    hw.ram_bandwidth_gbps = 100.0;  // default
    if (hw.unified_memory) {
        auto chip = gpu_bandwidth_for(hw.cpu_name);
        if (chip > 0) hw.ram_bandwidth_gbps = chip;
    }

    // GPU — on Apple Silicon, the GPU IS the chip. We expose it as a
    // pseudo-GPU with vram_bytes = 75% of RAM (matches the Python sizer).
    if (hw.unified_memory) {
        GPU g;
        g.name = hw.cpu_name;
        g.vendor = "apple";
        g.vram_bytes = static_cast<std::int64_t>(hw.ram_bytes * 0.75);
        g.bandwidth_gbps = gpu_bandwidth_for(hw.cpu_name);
        if (g.bandwidth_gbps == 0) g.bandwidth_gbps = 200;
        hw.gpus.push_back(std::move(g));
        hw.notes.push_back("Apple Silicon detected — GPU shares system RAM.");
    } else {
        hw.notes.push_back("Intel Mac — GPU detection limited.");
    }

    return hw;
}

} // namespace modelfit

#endif // __APPLE__
