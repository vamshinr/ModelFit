// Linux hardware detection: /proc/cpuinfo, /proc/meminfo, nvidia-smi for VRAM.
#if defined(__linux__)

#include "modelfit/hardware.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/utsname.h>
#include <unistd.h>

namespace modelfit {

// Read a key/value pair from /proc/meminfo.
static std::int64_t meminfo_value(const std::string& key) {
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            auto colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::int64_t kb = std::strtoll(line.c_str() + colon + 1, nullptr, 10);
            return kb * 1024;
        }
    }
    return 0;
}

// Parse /proc/cpuinfo for model name + physical core count.
static void parse_cpuinfo(HardwareProfile& hw) {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    int siblings = 0;
    int cores = 0;
    int processors = 0;
    while (std::getline(f, line)) {
        if (line.rfind("model name", 0) == 0 && hw.cpu_name.empty()) {
            auto colon = line.find(':');
            if (colon != std::string::npos)
                hw.cpu_name = line.substr(colon + 2);
        } else if (line.rfind("cpu cores", 0) == 0 && cores == 0) {
            auto colon = line.find(':');
            if (colon != std::string::npos)
                cores = std::atoi(line.c_str() + colon + 1);
        } else if (line.rfind("siblings", 0) == 0 && siblings == 0) {
            auto colon = line.find(':');
            if (colon != std::string::npos)
                siblings = std::atoi(line.c_str() + colon + 1);
        } else if (line.rfind("processor", 0) == 0) {
            processors++;
        } else if (line.rfind("cpu MHz", 0) == 0 && hw.cpu_freq_mhz == 0) {
            auto colon = line.find(':');
            if (colon != std::string::npos)
                hw.cpu_freq_mhz = std::strtod(line.c_str() + colon + 1, nullptr);
        }
    }
    hw.cpu_cores = cores > 0 ? cores : processors;
    hw.cpu_threads = siblings > 0 ? siblings : processors;
}

// Run nvidia-smi if available; one line per GPU.
static std::vector<GPU> probe_nvidia() {
    std::vector<GPU> out;
    FILE* pipe = popen(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null",
        "r");
    if (!pipe) return out;
    char buf[512];
    while (std::fgets(buf, sizeof(buf), pipe)) {
        std::string line(buf);
        auto comma = line.find(',');
        if (comma == std::string::npos) continue;
        std::string name = line.substr(0, comma);
        std::string memstr = line.substr(comma + 1);
        while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back()))) name.pop_back();
        std::int64_t mb = std::strtoll(memstr.c_str(), nullptr, 10);
        GPU g;
        g.name = name;
        g.vendor = "nvidia";
        g.vram_bytes = mb * MB;
        g.bandwidth_gbps = gpu_bandwidth_for(name);
        out.push_back(std::move(g));
    }
    pclose(pipe);
    return out;
}

HardwareProfile detect_hardware() {
    HardwareProfile hw;
    hw.platform = "Linux";

    struct utsname uts {};
    if (uname(&uts) == 0) hw.arch = uts.machine;

    parse_cpuinfo(hw);
    if (hw.cpu_name.empty()) hw.cpu_name = "Unknown CPU";

    hw.ram_bytes = meminfo_value("MemTotal");
    hw.ram_available_bytes = meminfo_value("MemAvailable");
    if (hw.ram_available_bytes == 0) hw.ram_available_bytes = hw.ram_bytes;

    // Bandwidth heuristic (matches Python). DDR5 ~80; DDR4 ~50.
    hw.ram_bandwidth_gbps = 50.0;
    std::string lower_cpu = hw.cpu_name;
    for (auto& c : lower_cpu) c = std::tolower(static_cast<unsigned char>(c));
    if (lower_cpu.find("13th gen") != std::string::npos
        || lower_cpu.find("14th gen") != std::string::npos
        || lower_cpu.find("core ultra") != std::string::npos
        || lower_cpu.find("ryzen 7000") != std::string::npos
        || lower_cpu.find("ryzen 8000") != std::string::npos
        || lower_cpu.find("ryzen 9000") != std::string::npos)
        hw.ram_bandwidth_gbps = 80.0;

    // GPUs
    hw.gpus = probe_nvidia();
    if (hw.gpus.empty()) hw.notes.push_back("No NVIDIA GPU detected (nvidia-smi missing).");

    return hw;
}

} // namespace modelfit

#endif // __linux__
