// Memory-fit walker — mirrors the Python `quantization.fit_model`.
#pragma once

#include "modelfit/types.h"

namespace modelfit {

struct FitResult {
    bool fits = false;
    std::string_view quant;     // empty when fits == false
    int context = 0;
    std::int64_t weights_bytes = 0;
    std::int64_t kv_bytes = 0;
    std::int64_t overhead_bytes = 0;
    std::int64_t total_bytes = 0;
    std::int64_t budget_bytes = 0;
    double quality_retention = 0;
    std::string tier;           // "gpu" | "cpu" | "unified"
    std::string reason;
};

// Walk Q8_0 → Q2_K, walk context max → 50% → 25% → 12.5% → min_context.
FitResult fit_model(const ModelSpec& m, const HardwareProfile& hw,
                    int min_context = 2048);

// Helpers (also used by reverse mode and the scorer).
std::int64_t weights_bytes(const ModelSpec& m, std::string_view quant);
std::int64_t kv_bytes_for(const ModelSpec& m, int context);
std::int64_t memory_required(const ModelSpec& m, std::string_view quant, int context);

constexpr double SAFETY_HEADROOM = 0.92;
constexpr double ACTIVATION_FRACTION = 0.05;
constexpr std::int64_t GPU_OVERHEAD_BYTES = 800LL * MB;
constexpr std::int64_t CPU_OVERHEAD_BYTES = 400LL * MB;

} // namespace modelfit
