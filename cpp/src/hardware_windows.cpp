// Windows hardware detection: GlobalMemoryStatusEx for RAM, registry / WMI
// for CPU, DXGI for GPU+VRAM enumeration.
#if defined(_WIN32)

#include "modelfit/hardware.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#include <dxgi.h>
#include <psapi.h>
#include <wbemidl.h>
#include <comdef.h>

#include <array>
#include <cstring>
#include <vector>

#pragma comment(lib, "dxgi.lib")

namespace modelfit {

static std::string wide_to_utf8(const wchar_t* w) {
    if (!w) return "";
    int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return "";
    std::string out(static_cast<size_t>(len - 1), 0);
    WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), len, nullptr, nullptr);
    return out;
}

static std::string cpu_brand_string() {
    int regs[4] = {0};
    std::array<char, 49> buf{};
    __cpuid(regs, 0x80000000);
    if (static_cast<unsigned>(regs[0]) < 0x80000004u) return "Unknown CPU";
    for (unsigned i = 0; i < 3; ++i) {
        __cpuid(regs, 0x80000002 + i);
        std::memcpy(buf.data() + i * 16, regs, 16);
    }
    std::string s(buf.data());
    while (!s.empty() && (s.front() == ' ' || s.front() == '\0')) s.erase(s.begin());
    while (!s.empty() && (s.back() == ' ' || s.back() == '\0')) s.pop_back();
    return s;
}

static std::vector<GPU> probe_dxgi_gpus() {
    std::vector<GPU> out;
    IDXGIFactory* factory = nullptr;
    if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&factory))))
        return out;
    IDXGIAdapter* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC desc{};
        if (SUCCEEDED(adapter->GetDesc(&desc))) {
            GPU g;
            g.name = wide_to_utf8(desc.Description);
            g.vram_bytes = static_cast<std::int64_t>(desc.DedicatedVideoMemory);
            // VendorId: NVIDIA=0x10DE, AMD=0x1002, Intel=0x8086
            if (desc.VendorId == 0x10DE) g.vendor = "nvidia";
            else if (desc.VendorId == 0x1002) g.vendor = "amd";
            else if (desc.VendorId == 0x8086) g.vendor = "intel";
            else g.vendor = "unknown";
            g.bandwidth_gbps = gpu_bandwidth_for(g.name);
            // Skip Microsoft Basic Render driver / integrated low-VRAM stubs.
            if (g.vram_bytes > 128 * MB) out.push_back(std::move(g));
        }
        adapter->Release();
    }
    factory->Release();
    return out;
}

HardwareProfile detect_hardware() {
    HardwareProfile hw;
    hw.platform = "Windows";
    SYSTEM_INFO si{};
    GetNativeSystemInfo(&si);
    switch (si.wProcessorArchitecture) {
        case PROCESSOR_ARCHITECTURE_AMD64: hw.arch = "x86_64"; break;
        case PROCESSOR_ARCHITECTURE_ARM64: hw.arch = "arm64"; break;
        case PROCESSOR_ARCHITECTURE_INTEL: hw.arch = "x86"; break;
        default: hw.arch = "unknown";
    }
    hw.cpu_threads = static_cast<int>(si.dwNumberOfProcessors);

    hw.cpu_name = cpu_brand_string();

    // Physical-core count via GetLogicalProcessorInformationEx.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len > 0) {
        std::vector<BYTE> buf(len);
        if (GetLogicalProcessorInformationEx(
                RelationProcessorCore,
                reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()),
                &len)) {
            int cores = 0;
            DWORD off = 0;
            while (off < len) {
                auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
                if (info->Relationship == RelationProcessorCore) cores++;
                off += info->Size;
            }
            hw.cpu_cores = cores;
        }
    }
    if (hw.cpu_cores == 0) hw.cpu_cores = hw.cpu_threads;

    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        hw.ram_bytes = static_cast<std::int64_t>(ms.ullTotalPhys);
        hw.ram_available_bytes = static_cast<std::int64_t>(ms.ullAvailPhys);
    }

    // Bandwidth heuristic.
    std::string lower = hw.cpu_name;
    for (auto& c : lower) c = std::tolower(static_cast<unsigned char>(c));
    hw.ram_bandwidth_gbps = 50.0;
    if (lower.find("13th gen") != std::string::npos
        || lower.find("14th gen") != std::string::npos
        || lower.find("core ultra") != std::string::npos
        || lower.find("ryzen 7000") != std::string::npos
        || lower.find("ryzen 8000") != std::string::npos
        || lower.find("ryzen 9000") != std::string::npos)
        hw.ram_bandwidth_gbps = 80.0;

    hw.gpus = probe_dxgi_gpus();
    if (hw.gpus.empty()) hw.notes.push_back("No dedicated GPU detected via DXGI.");
    hw.cpu_freq_mhz = 0;  // Reading nominal MHz from registry is noisy; leave 0.

    return hw;
}

} // namespace modelfit

#endif // _WIN32
