// Command-line entry point.
//
// Compiled to `modelfit-cli` (Windows: modelfit-cli.exe). Mirrors the Python
// `modelfit` CLI for the subset of subcommands that exist natively in C++.
//
// Supported subcommands:
//     inspect           Detected hardware (JSON optional)
//     rank              Rank models for this machine
//     info <model>      Show fit + score for one model
//     reverse <model>   Hardware required to reach a target tok/s
//     list              Dump the catalog
//     version           Print version
//
// Native, zero-dep, single binary.
#include "modelfit/hardware.h"
#include "modelfit/quantization.h"
#include "modelfit/scoring.h"
#include "modelfit/catalog_data.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace modelfit;

static constexpr const char* VERSION = "0.1.0";

static double to_gb(std::int64_t b) { return b / static_cast<double>(GB); }

static const ModelSpec* find_model(std::string_view q) {
    for (const auto& m : CATALOG)
        if (m.id == q || m.name == q) return &m;
    // Substring fallback (must be unique).
    const ModelSpec* match = nullptr;
    int count = 0;
    for (const auto& m : CATALOG) {
        if (std::string(m.id).find(q) != std::string::npos
            || std::string(m.name).find(q) != std::string::npos) {
            match = &m; count++;
        }
    }
    return count == 1 ? match : nullptr;
}

// -----------------------------------------------------------------------
// JSON emission — small helpers, no dep
// -----------------------------------------------------------------------
static void json_escape(std::ostream& os, std::string_view s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    os << buf;
                } else os << c;
        }
    }
    os << '"';
}

static void emit_hardware_json(std::ostream& os, const HardwareProfile& hw) {
    os << "{\n";
    os << "  \"cpu_name\": "; json_escape(os, hw.cpu_name); os << ",\n";
    os << "  \"cpu_cores\": " << hw.cpu_cores << ",\n";
    os << "  \"cpu_threads\": " << hw.cpu_threads << ",\n";
    os << "  \"cpu_freq_mhz\": " << hw.cpu_freq_mhz << ",\n";
    os << "  \"ram_gb\": " << to_gb(hw.ram_bytes) << ",\n";
    os << "  \"ram_available_gb\": " << to_gb(hw.ram_available_bytes) << ",\n";
    os << "  \"ram_bandwidth_gbps\": " << hw.ram_bandwidth_gbps << ",\n";
    os << "  \"platform\": "; json_escape(os, hw.platform); os << ",\n";
    os << "  \"arch\": "; json_escape(os, hw.arch); os << ",\n";
    os << "  \"unified_memory\": " << (hw.unified_memory ? "true" : "false") << ",\n";
    os << "  \"total_vram_gb\": " << to_gb(hw.total_vram_bytes()) << ",\n";
    os << "  \"effective_bandwidth_gbps\": " << hw.effective_bandwidth_gbps() << ",\n";
    os << "  \"gpus\": [\n";
    for (size_t i = 0; i < hw.gpus.size(); ++i) {
        const auto& g = hw.gpus[i];
        os << "    {";
        os << "\"name\": "; json_escape(os, g.name);
        os << ", \"vendor\": "; json_escape(os, g.vendor);
        os << ", \"vram_gb\": " << to_gb(g.vram_bytes);
        os << ", \"bandwidth_gbps\": " << g.bandwidth_gbps;
        os << "}" << (i + 1 < hw.gpus.size() ? "," : "") << "\n";
    }
    os << "  ]\n}";
}

// -----------------------------------------------------------------------
// Pretty table — printf-based, no Unicode dependence
// -----------------------------------------------------------------------
static void print_hardware(const HardwareProfile& hw) {
    std::printf("Hardware (%s %s)\n", hw.platform.c_str(), hw.arch.c_str());
    std::printf("  CPU: %s (%d cores / %d threads @ %.1f GHz)\n",
                hw.cpu_name.c_str(), hw.cpu_cores, hw.cpu_threads,
                hw.cpu_freq_mhz / 1000.0);
    std::printf("  RAM: %.1f GB (%.1f GB free, ~%.0f GB/s)\n",
                to_gb(hw.ram_bytes), to_gb(hw.ram_available_bytes),
                hw.ram_bandwidth_gbps);
    if (hw.gpus.empty()) {
        std::printf("  GPU: none — CPU inference only\n");
    } else {
        for (const auto& g : hw.gpus) {
            std::printf("  GPU: %s (%.1f GB VRAM, ~%.0f GB/s, %s)\n",
                        g.name.c_str(), to_gb(g.vram_bytes),
                        g.bandwidth_gbps, g.vendor.c_str());
        }
    }
    if (hw.unified_memory) std::printf("  Apple Silicon unified memory.\n");
}

static const char* tier_label(const std::string& t) {
    if (t == "gpu") return "GPU";
    if (t == "unified") return "Unified";
    return "CPU";
}

static void cmd_inspect(bool json) {
    auto hw = detect_hardware();
    if (json) { emit_hardware_json(std::cout, hw); std::cout << "\n"; }
    else      { print_hardware(hw); }
}

static void cmd_rank(std::string_view use_case, int min_ctx, int top, bool json) {
    auto hw = detect_hardware();
    auto ranked = rank_all(hw, use_case, min_ctx, /*include_unfit=*/false);
    if (json) {
        std::cout << "{\n  \"use_case\": "; json_escape(std::cout, use_case);
        std::cout << ",\n  \"hardware\": "; emit_hardware_json(std::cout, hw);
        std::cout << ",\n  \"fitting_models\": " << ranked.size();
        std::cout << ",\n  \"models\": [\n";
        int limit = std::min<int>(top, static_cast<int>(ranked.size()));
        for (int i = 0; i < limit; ++i) {
            const auto& s = ranked[i];
            std::cout << "    {";
            std::cout << "\"id\": "; json_escape(std::cout, s.model->id);
            std::cout << ", \"name\": "; json_escape(std::cout, s.model->name);
            std::cout << ", \"quant\": "; json_escape(std::cout, s.fit.quant);
            std::cout << ", \"context\": " << s.fit.context;
            std::cout << ", \"memory_gb\": " << to_gb(s.fit.total_bytes);
            std::cout << ", \"tokens_per_sec\": " << s.tokens_per_sec;
            std::cout << ", \"composite\": " << s.composite;
            std::cout << "}" << (i + 1 < limit ? "," : "") << "\n";
        }
        std::cout << "  ]\n}\n";
        return;
    }
    print_hardware(hw);
    std::printf("\nUse case: %s   Min context: %d tokens   "
                "%zu / %zu models fit.\n\n",
                std::string(use_case).c_str(), min_ctx, ranked.size(), CATALOG_SIZE);
    std::printf("  #  Score  Model                                Quant   Ctx       Mem   tok/s\n");
    std::printf("  -- -----  -----                                -----   ---       ---   -----\n");
    int limit = std::min<int>(top, static_cast<int>(ranked.size()));
    for (int i = 0; i < limit; ++i) {
        const auto& s = ranked[i];
        std::string name(s.model->name);
        if (name.size() > 35) name = name.substr(0, 32) + "...";
        std::string quant(s.fit.quant);
        std::printf("  %-2d %5.1f  %-35s  %-6s  %5dK  %5.1fG  %5.1f\n",
                    i + 1, s.composite, name.c_str(),
                    quant.c_str(),
                    s.fit.context / 1000, to_gb(s.fit.total_bytes),
                    s.tokens_per_sec);
    }
}

static void cmd_info(std::string_view query, std::string_view use_case,
                       int min_ctx, bool json) {
    const ModelSpec* m = find_model(query);
    if (!m) { std::fprintf(stderr, "Model not found: %.*s\n",
                              static_cast<int>(query.size()), query.data()); std::exit(2); }
    auto hw = detect_hardware();
    auto s = score_model(*m, hw, use_case, min_ctx);
    if (json) {
        std::cout << "{\n  \"id\": "; json_escape(std::cout, m->id);
        std::cout << ",\n  \"name\": "; json_escape(std::cout, m->name);
        std::cout << ",\n  \"family\": "; json_escape(std::cout, m->family);
        std::cout << ",\n  \"params_b\": " << m->params_b;
        std::cout << ",\n  \"context_max\": " << m->context_max;
        std::cout << ",\n  \"fits\": " << (s.fit.fits ? "true" : "false");
        if (s.fit.fits) {
            std::cout << ",\n  \"quant\": "; json_escape(std::cout, s.fit.quant);
            std::cout << ",\n  \"context\": " << s.fit.context;
            std::cout << ",\n  \"memory_gb\": " << to_gb(s.fit.total_bytes);
            std::cout << ",\n  \"tokens_per_sec\": " << s.tokens_per_sec;
            std::cout << ",\n  \"composite\": " << s.composite;
        }
        std::cout << "\n}\n";
        return;
    }
    print_hardware(hw);
    std::printf("\nModel: %s  (%s)\n", std::string(m->name).c_str(),
                std::string(m->id).c_str());
    std::printf("  Family: %s    Type: %s    Params: %.1fB (%.1fB active)\n",
                std::string(m->family).c_str(), std::string(m->type).c_str(),
                m->params_b, m->active_params_b);
    std::printf("  Max context: %d tokens.  Intrinsic quality: %.0f/100\n",
                m->context_max, m->quality);
    if (s.fit.fits) {
        std::printf("\n  [Fits] %s @ %d ctx -> %.1f GB on %s (budget %.1f GB)\n",
                    std::string(s.fit.quant).c_str(), s.fit.context,
                    to_gb(s.fit.total_bytes), tier_label(s.fit.tier),
                    to_gb(s.fit.budget_bytes));
        std::printf("  Composite: %.1f  (Q %.0f / S %.0f / C %.0f / Cap %.0f)\n",
                    s.composite, s.quality, s.speed, s.context, s.capability);
        std::printf("  Estimated speed: ~%.0f tokens/sec\n", s.tokens_per_sec);
    } else {
        std::printf("\n  [Won't fit] %s\n", s.fit.reason.c_str());
    }
}

static void cmd_list(bool json) {
    if (json) {
        std::cout << "[\n";
        for (size_t i = 0; i < CATALOG_SIZE; ++i) {
            const auto& m = CATALOG[i];
            std::cout << "  {";
            std::cout << "\"id\": "; json_escape(std::cout, m.id);
            std::cout << ", \"name\": "; json_escape(std::cout, m.name);
            std::cout << ", \"family\": "; json_escape(std::cout, m.family);
            std::cout << ", \"params_b\": " << m.params_b;
            std::cout << ", \"context_max\": " << m.context_max;
            std::cout << ", \"type\": "; json_escape(std::cout, m.type);
            std::cout << ", \"quality\": " << m.quality;
            std::cout << "}" << (i + 1 < CATALOG_SIZE ? "," : "") << "\n";
        }
        std::cout << "]\n";
        return;
    }
    std::printf("Catalog (%zu models):\n", CATALOG_SIZE);
    for (const auto& m : CATALOG) {
        std::string mn(m.name);
        if (mn.size() > 35) mn = mn.substr(0, 32) + "...";
        std::printf("  %-40s  %-10s  %5.1fB  %6dK  Q=%.0f\n",
                    std::string(m.id).c_str(), std::string(m.type).c_str(),
                    m.params_b, m.context_max / 1000, m.quality);
    }
}

// Tiny argv helper
static bool has_flag(int argc, char** argv, std::string_view f) {
    for (int i = 0; i < argc; ++i) if (argv[i] == f) return true;
    return false;
}
static std::string_view arg_value(int argc, char** argv, std::string_view name,
                                     std::string_view def) {
    for (int i = 0; i < argc - 1; ++i)
        if (argv[i] == name) return argv[i + 1];
    return def;
}

static void usage() {
    std::printf(
        "modelfit-cli %s — auto-detect hardware and rank %zu LLMs.\n"
        "\n"
        "Usage:\n"
        "  modelfit-cli [inspect|rank|info|list|version] [options]\n"
        "\n"
        "Common options:\n"
        "  --json                Machine-readable JSON output\n"
        "  --use-case <X>        chat|reasoning|code|math|long-context|agent|research|balanced\n"
        "  --min-context <N>     Minimum acceptable context tokens (default 2048)\n"
        "  --top <N>             How many to show (default 15)\n"
        "\n"
        "Examples:\n"
        "  modelfit-cli                              # rank for this machine\n"
        "  modelfit-cli inspect                      # detected hardware\n"
        "  modelfit-cli rank --use-case reasoning    # reasoning-leaning weights\n"
        "  modelfit-cli info llama-3.1-8b-instruct   # detail for one model\n"
        "  modelfit-cli list --json                  # full catalog as JSON\n",
        VERSION, CATALOG_SIZE);
}

int main(int argc, char** argv) {
    if (argc >= 2 && (std::string_view(argv[1]) == "--version"
                         || std::string_view(argv[1]) == "-V")) {
        std::printf("modelfit-cli %s  (catalog: %zu models)\n", VERSION, CATALOG_SIZE);
        return 0;
    }
    if (argc >= 2 && (std::string_view(argv[1]) == "--help"
                         || std::string_view(argv[1]) == "-h")) {
        usage(); return 0;
    }

    // Default command = rank if first arg starts with '-' or no command.
    int cmd_idx = 1;
    std::string_view cmd = (argc < 2 || argv[1][0] == '-') ? "rank" : argv[1];
    if (cmd != "rank") cmd_idx = 2;
    bool json = has_flag(argc, argv, "--json");
    auto use_case = arg_value(argc, argv, "--use-case", "balanced");
    auto min_ctx_s = arg_value(argc, argv, "--min-context", "2048");
    auto top_s = arg_value(argc, argv, "--top", "15");
    int min_ctx = std::atoi(std::string(min_ctx_s).c_str());
    int top = std::atoi(std::string(top_s).c_str());

    if (cmd == "inspect") cmd_inspect(json);
    else if (cmd == "rank") cmd_rank(use_case, min_ctx, top, json);
    else if (cmd == "list") cmd_list(json);
    else if (cmd == "info") {
        if (argc <= cmd_idx) { std::fprintf(stderr, "info: missing model id\n"); return 2; }
        cmd_info(argv[cmd_idx], use_case, min_ctx, json);
    } else if (cmd == "version") {
        std::printf("modelfit-cli %s\n", VERSION);
    } else {
        usage(); return 1;
    }
    return 0;
}
