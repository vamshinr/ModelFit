#include "modelfit/quantization.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace modelfit {

std::int64_t weights_bytes(const ModelSpec& m, std::string_view quant) {
    const QuantInfo* q = find_quant(quant);
    if (!q) return 0;
    return static_cast<std::int64_t>(m.params_b * 1'000'000'000.0 * q->bytes_per_param);
}

std::int64_t kv_bytes_for(const ModelSpec& m, int context) {
    return static_cast<std::int64_t>(m.kv_bytes_per_token) * static_cast<std::int64_t>(context);
}

std::int64_t memory_required(const ModelSpec& m, std::string_view quant, int context) {
    auto w = weights_bytes(m, quant);
    auto kv = kv_bytes_for(m, context);
    auto activations = static_cast<std::int64_t>((w + kv) * ACTIVATION_FRACTION);
    return w + kv + activations;
}

struct TierInfo { std::int64_t budget; std::string tier; std::int64_t overhead; };

static TierInfo budget_for(const HardwareProfile& hw) {
    if (hw.unified_memory) {
        std::int64_t raw = static_cast<std::int64_t>(hw.ram_bytes * 0.75) - GPU_OVERHEAD_BYTES;
        return {raw, "unified", GPU_OVERHEAD_BYTES};
    }
    std::int64_t vram = 0;
    for (const auto& g : hw.gpus) vram += g.vram_bytes;
    if (vram > 0) return {vram - GPU_OVERHEAD_BYTES, "gpu", GPU_OVERHEAD_BYTES};
    return {hw.ram_bytes - CPU_OVERHEAD_BYTES, "cpu", CPU_OVERHEAD_BYTES};
}

FitResult fit_model(const ModelSpec& m, const HardwareProfile& hw, int min_context) {
    auto t = budget_for(hw);
    std::int64_t budget = static_cast<std::int64_t>(t.budget * SAFETY_HEADROOM);

    // Context fallbacks: 100%, 50%, 25%, 12.5%, plus the explicit floor.
    std::vector<int> contexts;
    for (double frac : {1.0, 0.5, 0.25, 0.125}) {
        int c = static_cast<int>(m.context_max * frac);
        if (c >= std::max(min_context, 512)) {
            if (std::find(contexts.begin(), contexts.end(), c) == contexts.end())
                contexts.push_back(c);
        }
    }
    int floor = std::max(min_context, 512);
    if (std::find(contexts.begin(), contexts.end(), floor) == contexts.end()
        && floor < m.context_max)
        contexts.push_back(floor);

    FitResult best_fail;
    best_fail.budget_bytes = budget;
    best_fail.overhead_bytes = t.overhead;
    best_fail.tier = t.tier;

    for (auto q : QUANT_ORDER) {
        const QuantInfo* qi = find_quant(q);
        if (!qi) continue;
        for (int ctx : contexts) {
            auto total = memory_required(m, q, ctx) + t.overhead;
            if (total <= budget) {
                FitResult r;
                r.fits = true;
                r.quant = q;
                r.context = ctx;
                r.weights_bytes = weights_bytes(m, q);
                r.kv_bytes = kv_bytes_for(m, ctx);
                r.overhead_bytes = t.overhead;
                r.total_bytes = total;
                r.budget_bytes = budget;
                r.quality_retention = qi->quality_retention;
                r.tier = t.tier;
                std::ostringstream os;
                os << "fits at " << q << " with " << ctx
                   << " ctx (" << (total / 1024.0 / 1024 / 1024) << " GB <= "
                   << (budget / 1024.0 / 1024 / 1024) << " GB " << t.tier << ")";
                r.reason = os.str();
                return r;
            }
            best_fail.fits = false;
            best_fail.quant = q;
            best_fail.context = ctx;
            best_fail.weights_bytes = weights_bytes(m, q);
            best_fail.kv_bytes = kv_bytes_for(m, ctx);
            best_fail.total_bytes = total;
            best_fail.quality_retention = qi->quality_retention;
        }
    }
    std::ostringstream os;
    os << "won't fit even at Q2_K — needs ~" << (best_fail.total_bytes / 1024.0 / 1024 / 1024)
       << " GB, have " << (budget / 1024.0 / 1024 / 1024) << " GB " << t.tier;
    best_fail.reason = os.str();
    return best_fail;
}

} // namespace modelfit
