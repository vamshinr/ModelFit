#include "modelfit/types.h"
#include "modelfit/hardware.h"

#include <algorithm>
#include <unordered_map>

namespace modelfit {

const QuantInfo* find_quant(std::string_view code) {
    for (const auto& q : QUANT_TABLE)
        if (q.code == code) return &q;
    return nullptr;
}

std::int64_t HardwareProfile::total_vram_bytes() const {
    if (unified_memory && !gpus.empty())
        return static_cast<std::int64_t>(ram_bytes * 0.75);
    std::int64_t sum = 0;
    for (const auto& g : gpus) sum += g.vram_bytes;
    return sum;
}

double HardwareProfile::effective_bandwidth_gbps() const {
    if (!gpus.empty()) {
        double best = 0;
        for (const auto& g : gpus) best = std::max(best, g.bandwidth_gbps);
        return best;
    }
    return ram_bandwidth_gbps;
}

const GPU* HardwareProfile::best_gpu() const {
    if (gpus.empty()) return nullptr;
    const GPU* best = &gpus.front();
    for (const auto& g : gpus)
        if (g.vram_bytes > best->vram_bytes) best = &g;
    return best;
}

// ---------------------------------------------------------------------------
// GPU bandwidth lookup table — kept in sync with modelfit/hardware.py.
// ---------------------------------------------------------------------------
static const std::unordered_map<std::string, double> GPU_BANDWIDTH = {
    // NVIDIA consumer
    {"rtx 5090", 1792}, {"rtx 5080", 960}, {"rtx 5070 ti", 896}, {"rtx 5070", 672},
    {"rtx 4090", 1008}, {"rtx 4080 super", 736}, {"rtx 4080", 717},
    {"rtx 4070 ti super", 672}, {"rtx 4070 ti", 504}, {"rtx 4070 super", 504},
    {"rtx 4070", 504}, {"rtx 4060 ti", 288}, {"rtx 4060", 272},
    {"rtx 3090 ti", 1008}, {"rtx 3090", 936}, {"rtx 3080 ti", 912},
    {"rtx 3080", 760}, {"rtx 3070 ti", 608}, {"rtx 3070", 448},
    {"rtx 3060 ti", 448}, {"rtx 3060", 360},
    {"rtx 2080 ti", 616}, {"rtx 2080 super", 496}, {"rtx 2080", 448},
    {"rtx 2070 super", 448}, {"rtx 2070", 448},
    {"rtx 2060 super", 448}, {"rtx 2060", 336},
    // NVIDIA data-center
    {"h200", 4800}, {"h100", 3350}, {"a100", 2039}, {"l40s", 864}, {"l40", 864},
    {"a40", 696}, {"a10", 600}, {"a6000", 768}, {"a5000", 768}, {"a4000", 448},
    {"v100", 900}, {"t4", 320},
    // AMD
    {"rx 7900 xtx", 960}, {"rx 7900 xt", 800}, {"rx 7800 xt", 624},
    {"rx 6900 xt", 512}, {"rx 6800 xt", 512}, {"rx 6700 xt", 384},
    {"mi300x", 5300}, {"mi250x", 3276}, {"mi210", 1638},
    // Apple unified
    {"m1", 68}, {"m1 pro", 200}, {"m1 max", 400}, {"m1 ultra", 800},
    {"m2", 100}, {"m2 pro", 200}, {"m2 max", 400}, {"m2 ultra", 800},
    {"m3", 100}, {"m3 pro", 150}, {"m3 max", 400},
    {"m4", 120}, {"m4 pro", 273}, {"m4 max", 546},
    // Intel
    {"arc a770", 560}, {"arc a750", 512}, {"arc b580", 456},
};

static std::string lowercased(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c){ return std::tolower(c); });
    return s;
}

double gpu_bandwidth_for(const std::string& name) {
    auto n = lowercased(name);
    // Longest-key-first match: H100, M2 Ultra, "RTX 4080 Super" should
    // beat their shorter prefixes.
    std::vector<std::pair<std::string, double>> sorted_keys(
        GPU_BANDWIDTH.begin(), GPU_BANDWIDTH.end());
    std::sort(sorted_keys.begin(), sorted_keys.end(),
                [](const auto& a, const auto& b){ return a.first.size() > b.first.size(); });
    for (const auto& [k, v] : sorted_keys) {
        if (n.find(k) != std::string::npos) return v;
    }
    if (n.find("h100") != std::string::npos) return 3350;
    if (n.find("a100") != std::string::npos) return 2039;
    if (n.find("rtx 40") != std::string::npos) return 600;
    if (n.find("rtx 30") != std::string::npos) return 500;
    if (n.find("rtx") != std::string::npos)    return 350;
    if (n.find("apple") != std::string::npos)  return 200;
    return 250;
}

} // namespace modelfit
