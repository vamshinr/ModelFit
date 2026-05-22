// Composite scoring + ranking.
#pragma once

#include "modelfit/quantization.h"
#include <array>
#include <string>
#include <vector>

namespace modelfit {

struct ScoredModel {
    const ModelSpec* model = nullptr;
    FitResult fit;
    double tokens_per_sec = 0;
    double quality = 0;
    double speed = 0;
    double context = 0;
    double capability = 0;
    double composite = 0;
    std::string use_case;
};

// 8 use cases. Weights mirror the Python `USE_CASE_WEIGHTS`.
struct UseCaseWeights {
    std::string_view name;
    double quality, speed, context, capability;
};
inline constexpr std::array<UseCaseWeights, 8> USE_CASES = {{
    {"chat",         0.30, 0.35, 0.15, 0.20},
    {"reasoning",    0.50, 0.15, 0.20, 0.15},
    {"code",         0.35, 0.25, 0.20, 0.20},
    {"math",         0.50, 0.15, 0.15, 0.20},
    {"long-context", 0.25, 0.15, 0.45, 0.15},
    {"agent",        0.35, 0.25, 0.25, 0.15},
    {"research",     0.40, 0.20, 0.25, 0.15},
    {"balanced",     0.30, 0.30, 0.20, 0.20},
}};

const UseCaseWeights* find_use_case(std::string_view name);

double estimate_tokens_per_sec(const ModelSpec& m, const FitResult& fit,
                                const HardwareProfile& hw);

ScoredModel score_model(const ModelSpec& m, const HardwareProfile& hw,
                          std::string_view use_case = "balanced",
                          int min_context = 2048);

std::vector<ScoredModel> rank_all(const HardwareProfile& hw,
                                    std::string_view use_case = "balanced",
                                    int min_context = 2048,
                                    bool include_unfit = false);

} // namespace modelfit
