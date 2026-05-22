#include "modelfit/scoring.h"
#include "modelfit/catalog_data.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace modelfit {

const UseCaseWeights* find_use_case(std::string_view name) {
    for (const auto& w : USE_CASES) if (w.name == name) return &w;
    return nullptr;
}

constexpr double BANDWIDTH_UTILIZATION = 0.65;
constexpr double MOE_SPEEDUP = 1.7;
constexpr double MIN_TPS = 0.3;
constexpr double MAX_TPS = 250.0;

double estimate_tokens_per_sec(const ModelSpec& m, const FitResult& fit,
                                const HardwareProfile& hw) {
    if (!fit.fits) return 0;
    double active_ratio = (m.params_b > 0) ? (m.active_params_b / m.params_b) : 1.0;
    double active_weight_bytes = static_cast<double>(fit.weights_bytes) * active_ratio;
    double bytes_per_step = active_weight_bytes + (fit.kv_bytes * 0.1);
    double bandwidth = hw.effective_bandwidth_gbps() * 1024.0 * 1024 * 1024 * BANDWIDTH_UTILIZATION;
    if (bytes_per_step <= 0) return 0;
    double tps = bandwidth / bytes_per_step;
    if (active_ratio < 0.99) tps *= MOE_SPEEDUP;
    return std::max(MIN_TPS, std::min(MAX_TPS, tps));
}

static double quality_subscore(const ModelSpec& m, const FitResult& fit) {
    return std::clamp(m.quality * fit.quality_retention, 0.0, 100.0);
}

static double speed_subscore(double tps) {
    if (tps <= 0) return 0;
    double norm = (std::log10(tps) - std::log10(MIN_TPS))
                / (std::log10(MAX_TPS) - std::log10(MIN_TPS));
    return std::clamp(norm * 100.0, 0.0, 100.0);
}

static double context_subscore(int fit_context, int model_max) {
    if (fit_context <= 0) return 0;
    double abs_score = std::min(100.0,
        std::log2(std::max(2048, fit_context) / 2048.0) * 18.0);
    double frac = model_max > 0 ? std::min(1.0, static_cast<double>(fit_context) / model_max) : 1.0;
    return abs_score * (0.6 + 0.4 * frac);
}

static double capability_subscore(const ModelSpec& m, std::string_view use_case) {
    auto t = m.type;
    int score = 60;
    auto match = [&](std::string_view uc) { return uc == use_case; };
    if (match("chat")) {
        if (t == "chat") score = 95; else if (t == "instruct") score = 88;
        else if (t == "reasoning") score = 80; else if (t == "code") score = 60;
        else if (t == "math") score = 65; else if (t == "vision") score = 75;
        else if (t == "base") score = 50;
    } else if (match("reasoning")) {
        if (t == "reasoning") score = 100; else if (t == "instruct") score = 75;
        else if (t == "chat") score = 70; else if (t == "math") score = 80;
        else if (t == "code") score = 65; else if (t == "base") score = 40;
        else if (t == "vision") score = 60;
    } else if (match("code")) {
        if (t == "code") score = 100; else if (t == "instruct") score = 78;
        else if (t == "chat") score = 65; else if (t == "reasoning") score = 80;
        else if (t == "math") score = 65; else if (t == "base") score = 50;
        else if (t == "vision") score = 55;
    } else if (match("math")) {
        if (t == "math") score = 100; else if (t == "reasoning") score = 90;
        else if (t == "instruct") score = 70; else if (t == "chat") score = 65;
        else if (t == "code") score = 70; else if (t == "base") score = 40;
        else if (t == "vision") score = 50;
    } else if (match("long-context")) {
        if (t == "instruct") score = 85; else if (t == "chat") score = 82;
        else if (t == "code") score = 80; else if (t == "reasoning") score = 80;
        else if (t == "math") score = 60; else if (t == "base") score = 55;
        else if (t == "vision") score = 70;
    } else if (match("agent")) {
        if (t == "instruct") score = 95; else if (t == "chat") score = 88;
        else if (t == "code") score = 80; else if (t == "reasoning") score = 85;
        else if (t == "math") score = 60; else if (t == "base") score = 50;
        else if (t == "vision") score = 65;
    } else if (match("research")) {
        if (t == "reasoning") score = 90; else if (t == "instruct") score = 85;
        else if (t == "chat") score = 80; else if (t == "math") score = 80;
        else if (t == "code") score = 70; else if (t == "base") score = 60;
        else if (t == "vision") score = 65;
    } else {  // balanced
        if (t == "instruct") score = 88; else if (t == "chat") score = 85;
        else if (t == "code") score = 75; else if (t == "reasoning") score = 80;
        else if (t == "math") score = 70; else if (t == "base") score = 55;
        else if (t == "vision") score = 70;
    }

    // Tag boosts
    std::unordered_set<std::string_view> tags;
    for (auto t : m.tags) tags.insert(t);
    if (match("code") && tags.count("code")) score = std::min(100, score + 5);
    if (match("reasoning") && tags.count("reasoning")) score = std::min(100, score + 5);
    if (match("math") && tags.count("math")) score = std::min(100, score + 5);
    return static_cast<double>(score);
}

ScoredModel score_model(const ModelSpec& m, const HardwareProfile& hw,
                          std::string_view use_case, int min_context) {
    const auto* w = find_use_case(use_case);
    if (!w) w = find_use_case("balanced");

    FitResult fit = fit_model(m, hw, min_context);
    ScoredModel s;
    s.model = &m;
    s.fit = fit;
    s.use_case = std::string(use_case);
    if (!fit.fits) return s;
    s.tokens_per_sec = estimate_tokens_per_sec(m, fit, hw);
    s.quality = quality_subscore(m, fit);
    s.speed = speed_subscore(s.tokens_per_sec);
    s.context = context_subscore(fit.context, m.context_max);
    s.capability = capability_subscore(m, use_case);
    s.composite = w->quality * s.quality + w->speed * s.speed
                + w->context * s.context + w->capability * s.capability;
    return s;
}

std::vector<ScoredModel> rank_all(const HardwareProfile& hw,
                                    std::string_view use_case,
                                    int min_context, bool include_unfit) {
    std::vector<ScoredModel> out;
    out.reserve(CATALOG_SIZE);
    for (const auto& m : CATALOG) {
        auto s = score_model(m, hw, use_case, min_context);
        if (!include_unfit && !s.fit.fits) continue;
        out.push_back(std::move(s));
    }
    std::sort(out.begin(), out.end(),
                [](const ScoredModel& a, const ScoredModel& b){
                    return a.composite > b.composite;
                });
    return out;
}

} // namespace modelfit
