// modelfit core types (mirror of the Python ModelSpec / HardwareProfile).
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace modelfit {

// Quantization codes match the Python QUANT_ORDER exactly.
inline constexpr std::array<std::string_view, 8> QUANT_ORDER = {
    "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q3_K_S", "Q2_K",
};

struct QuantInfo {
    std::string_view code;
    double bytes_per_param;
    double quality_retention;
    std::string_view label;
    std::string_view stars;
};

inline constexpr std::array<QuantInfo, 9> QUANT_TABLE = {{
    {"F16",    2.00, 1.000, "Full precision",     "*****"},
    {"Q8_0",   1.06, 0.998, "Excellent quality",  "*****"},
    {"Q6_K",   0.82, 0.992, "Great quality",      "****+"},
    {"Q5_K_M", 0.71, 0.985, "Great quality",      "****+"},
    {"Q4_K_M", 0.59, 0.965, "Good quality",       "****"},
    {"Q4_K_S", 0.56, 0.955, "Good quality",       "****"},
    {"Q3_K_M", 0.49, 0.910, "Compressed",         "***"},
    {"Q3_K_S", 0.45, 0.880, "Heavily compressed", "**+"},
    {"Q2_K",   0.40, 0.820, "Heavily compressed", "**"},
}};

const QuantInfo* find_quant(std::string_view code);


// ---- Model catalog entry --------------------------------------------------
struct ModelSpec {
    std::string_view id;
    std::string_view name;
    std::string_view family;
    double params_b;
    double active_params_b;
    int context_max;
    std::string_view type;
    double quality;             // 0-100 intrinsic at F16
    std::int64_t kv_bytes_per_token;
    std::vector<std::string_view> tags;
};


// ---- Hardware profile ----------------------------------------------------
struct GPU {
    std::string name;
    std::string vendor;         // "nvidia" | "amd" | "apple" | "intel"
    std::int64_t vram_bytes = 0;
    double bandwidth_gbps = 0;  // peak advertised
};

struct HardwareProfile {
    std::string cpu_name;
    int cpu_cores = 0;
    int cpu_threads = 0;
    double cpu_freq_mhz = 0;
    std::int64_t ram_bytes = 0;
    std::int64_t ram_available_bytes = 0;
    double ram_bandwidth_gbps = 0;
    std::string platform;       // "Darwin" | "Linux" | "Windows"
    std::string arch;
    bool unified_memory = false;
    std::vector<GPU> gpus;
    std::vector<std::string> notes;

    std::int64_t total_vram_bytes() const;
    double effective_bandwidth_gbps() const;
    const GPU* best_gpu() const;
};

constexpr std::int64_t GB = 1024LL * 1024 * 1024;
constexpr std::int64_t MB = 1024LL * 1024;

} // namespace modelfit
