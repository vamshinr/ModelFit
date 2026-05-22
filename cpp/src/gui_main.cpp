// modelfit-gui — a Dear ImGui-based desktop app.
//
// Three tabs:
//   1. Models       — ranked table, filters, click for details
//   2. Hardware     — detected hardware panel
//   3. Glossary     — full term list + searchable + category sidebar
//
// Built with `cmake -DMODELFIT_GUI=ON .. && cmake --build .`
#include "modelfit/hardware.h"
#include "modelfit/scoring.h"
#include "modelfit/quantization.h"
#include "modelfit/catalog_data.h"
#include "modelfit/glossary.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#define GL_SILENCE_DEPRECATION
#if defined(__APPLE__)
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace modelfit;

static double to_gb(std::int64_t b) { return b / static_cast<double>(GB); }

static const char* UI_USE_CASES[] = {
    "balanced", "chat", "reasoning", "code", "math",
    "long-context", "agent", "research"
};
static const int N_UI_USE_CASES = static_cast<int>(sizeof(UI_USE_CASES) / sizeof(UI_USE_CASES[0]));

static void glfw_error(int, const char* desc) {
    std::fprintf(stderr, "GLFW error: %s\n", desc);
}

// -----------------------------------------------------------------------
// Font + DPI + user-zoom setup
//
// The HiDPI story for ImGui:
//
//   * `io.DisplaySize` is set by the GLFW backend in *logical* points.
//   * `io.DisplayFramebufferScale` is the framebuffer:window ratio (2.0 on
//      Retina). The OpenGL backend uses it to size the GL viewport.
//   * ImGui draws everything in logical points.
//
// So to get crisp text without making the UI take 4× the screen:
//
//   1. Load the font at  base_pt × dpi × user_zoom  texture pixels.
//   2. Set `io.FontGlobalScale = 1/dpi` so the *rendered* size in logical
//      points stays at `base_pt × user_zoom`.
//   3. Apply `ScaleAllSizes(user_zoom)` to a saved baseline style (so
//      paddings track the user zoom, but *not* the OS DPI).
//
// Previously we double-counted DPI: ScaleAllSizes(2.0) on Retina made
// everything massive. This rewrite separates the two scales cleanly.
// -----------------------------------------------------------------------
static bool file_exists(const char* path) {
    std::ifstream f(path);
    return f.good();
}

static const char* k_regular_candidates[] = {
#if defined(__APPLE__)
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Geneva.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
#elif defined(_WIN32)
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/verdana.ttf",
#else
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
#endif
    nullptr,
};

struct ZoomState {
    float dpi_scale = 1.0f;        // OS-reported (2.0 on Retina). Read-only.
    float user_zoom = 1.0f;        // User's zoom multiplier — adjustable.
    bool  fonts_dirty = true;      // Set to rebuild atlas + re-apply style.
    ImGuiStyle baseline;           // Saved before any scaling — source of truth.
    const char* font_path = nullptr;
    static constexpr float MIN_ZOOM = 0.70f;
    static constexpr float MAX_ZOOM = 2.00f;
    static constexpr float STEP = 0.10f;
};

static void apply_palette(ImGuiStyle& s) {
    s.WindowRounding  = 4.0f;
    s.FrameRounding   = 4.0f;
    s.GrabRounding    = 4.0f;
    s.PopupRounding   = 4.0f;
    s.TabRounding     = 4.0f;
    s.WindowBorderSize = 0;
    s.WindowPadding   = ImVec2(12, 10);
    s.FramePadding    = ImVec2(8, 5);
    s.ItemSpacing     = ImVec2(8, 6);
    s.ItemInnerSpacing= ImVec2(6, 5);
    s.ScrollbarSize   = 14;
    s.GrabMinSize     = 12;

    auto& colors = s.Colors;
    colors[ImGuiCol_WindowBg]        = ImVec4(0.08f, 0.09f, 0.11f, 1.00f);
    colors[ImGuiCol_ChildBg]         = ImVec4(0.10f, 0.11f, 0.13f, 1.00f);
    colors[ImGuiCol_PopupBg]         = ImVec4(0.11f, 0.12f, 0.14f, 0.98f);
    colors[ImGuiCol_Border]          = ImVec4(0.20f, 0.22f, 0.26f, 0.50f);
    colors[ImGuiCol_FrameBg]         = ImVec4(0.16f, 0.18f, 0.22f, 0.85f);
    colors[ImGuiCol_FrameBgHovered]  = ImVec4(0.22f, 0.26f, 0.32f, 0.85f);
    colors[ImGuiCol_FrameBgActive]   = ImVec4(0.26f, 0.32f, 0.40f, 0.85f);
    colors[ImGuiCol_TitleBg]         = ImVec4(0.10f, 0.11f, 0.13f, 1.00f);
    colors[ImGuiCol_Header]          = ImVec4(0.22f, 0.40f, 0.66f, 0.55f);
    colors[ImGuiCol_HeaderHovered]   = ImVec4(0.30f, 0.50f, 0.78f, 0.65f);
    colors[ImGuiCol_HeaderActive]    = ImVec4(0.36f, 0.58f, 0.88f, 0.75f);
    colors[ImGuiCol_Button]          = ImVec4(0.22f, 0.40f, 0.66f, 0.55f);
    colors[ImGuiCol_ButtonHovered]   = ImVec4(0.30f, 0.50f, 0.78f, 0.65f);
    colors[ImGuiCol_ButtonActive]    = ImVec4(0.36f, 0.58f, 0.88f, 0.75f);
    colors[ImGuiCol_Tab]             = ImVec4(0.13f, 0.15f, 0.18f, 1.00f);
    colors[ImGuiCol_TabHovered]      = ImVec4(0.32f, 0.50f, 0.78f, 0.80f);
    colors[ImGuiCol_TabActive]       = ImVec4(0.22f, 0.40f, 0.66f, 1.00f);
    colors[ImGuiCol_TabUnfocused]    = ImVec4(0.10f, 0.11f, 0.13f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.16f, 0.20f, 0.26f, 1.00f);
    colors[ImGuiCol_Text]            = ImVec4(0.92f, 0.93f, 0.94f, 1.00f);
    colors[ImGuiCol_TextDisabled]    = ImVec4(0.55f, 0.58f, 0.62f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]   = ImVec4(0.16f, 0.18f, 0.22f, 1.00f);
    colors[ImGuiCol_TableRowBg]      = ImVec4(0.10f, 0.11f, 0.13f, 1.00f);
    colors[ImGuiCol_TableRowBgAlt]   = ImVec4(0.13f, 0.14f, 0.16f, 1.00f);
    colors[ImGuiCol_Separator]       = ImVec4(0.20f, 0.22f, 0.26f, 0.60f);
    colors[ImGuiCol_CheckMark]       = ImVec4(0.45f, 0.78f, 0.55f, 1.00f);
    colors[ImGuiCol_PlotHistogram]   = ImVec4(0.45f, 0.78f, 0.55f, 1.00f);
}

// Build the font atlas + apply the style at the current zoom/dpi.
// Call once at startup, and after every change to user_zoom.
static void rebuild_fonts(ZoomState& zs) {
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();

    const float base_pt = 15.0f;
    const float font_px = base_pt * zs.dpi_scale * zs.user_zoom;

    ImFontConfig cfg;
    cfg.OversampleH = 3;
    cfg.OversampleV = 1;
    cfg.PixelSnapH = false;

    if (!zs.font_path) {
        for (int i = 0; k_regular_candidates[i]; ++i) {
            if (file_exists(k_regular_candidates[i])) {
                zs.font_path = k_regular_candidates[i];
                break;
            }
        }
    }

    bool loaded = false;
    if (zs.font_path) {
        if (io.Fonts->AddFontFromFileTTF(zs.font_path, font_px, &cfg)) {
            loaded = true;
        }
    }
    if (!loaded) {
        ImFontConfig dcfg;
        dcfg.SizePixels = font_px;
        io.Fonts->AddFontDefault(&dcfg);
    }

    // Render at logical points: shrink by DPI but NOT by user_zoom — user_zoom
    // is already baked into the texture size, so it appears bigger.
    io.FontGlobalScale = 1.0f / zs.dpi_scale;

    // Reset style to baseline, then scale only by user_zoom.
    ImGui::GetStyle() = zs.baseline;
    if (zs.user_zoom != 1.0f) {
        ImGui::GetStyle().ScaleAllSizes(zs.user_zoom);
    }
}

static void setup_initial_style(ZoomState& zs, GLFWwindow* window) {
    ImGuiStyle baseline;
    ImGui::StyleColorsDark(&baseline);
    apply_palette(baseline);
    zs.baseline = baseline;

    float xscale = 1.0f, yscale = 1.0f;
    glfwGetWindowContentScale(window, &xscale, &yscale);
    zs.dpi_scale = std::max({xscale, yscale, 1.0f});
    zs.user_zoom = 1.0f;
    zs.fonts_dirty = false;
    rebuild_fonts(zs);
    std::fprintf(stderr, "[modelfit-gui] dpi_scale=%.2f, font=%s\n",
                  zs.dpi_scale, zs.font_path ? zs.font_path : "(ImGui default)");
}

static void zoom_set(ZoomState& zs, float v) {
    v = std::clamp(v, ZoomState::MIN_ZOOM, ZoomState::MAX_ZOOM);
    if (std::abs(v - zs.user_zoom) > 0.001f) {
        zs.user_zoom = v;
        zs.fonts_dirty = true;
    }
}
static void zoom_in(ZoomState& zs)    { zoom_set(zs, zs.user_zoom + ZoomState::STEP); }
static void zoom_out(ZoomState& zs)   { zoom_set(zs, zs.user_zoom - ZoomState::STEP); }
static void zoom_reset(ZoomState& zs) { zoom_set(zs, 1.0f); }

// -----------------------------------------------------------------------
// Application state
// -----------------------------------------------------------------------
struct AppState {
    HardwareProfile hw;
    std::vector<ScoredModel> ranked;
    char filter[128] = "";
    int use_case_idx = 0;       // balanced
    int min_ctx_idx = 1;        // 4096
    int selected_model = -1;
    std::string glossary_search;
    int selected_category = -1; // -1 = "all"
    int selected_term = -1;     // index into filtered term list

    static const int CTX_VALUES[6];
};
const int AppState::CTX_VALUES[6] = {2048, 4096, 8192, 16384, 32768, 131072};
static const char* CTX_LABELS[6] = {
    "2K", "4K", "8K", "16K", "32K", "128K"
};

static void recompute(AppState& s) {
    s.ranked = rank_all(s.hw, UI_USE_CASES[s.use_case_idx],
                          AppState::CTX_VALUES[s.min_ctx_idx], false);
}

// -----------------------------------------------------------------------
// Tab: Models
// -----------------------------------------------------------------------
static const char* tier_label(const std::string& t) {
    if (t == "gpu") return "GPU";
    if (t == "unified") return "Unified";
    return "CPU";
}

static void draw_models_tab(AppState& s) {
    ImGui::Spacing();
    ImGui::Text("Optimise for:");
    ImGui::SameLine();
    if (ImGui::Combo("##use_case", &s.use_case_idx, UI_USE_CASES, N_UI_USE_CASES))
        recompute(s);
    ImGui::SameLine(); ImGui::Text("   Min context:");
    ImGui::SameLine();
    if (ImGui::Combo("##min_ctx", &s.min_ctx_idx, CTX_LABELS,
                       static_cast<int>(sizeof(CTX_LABELS) / sizeof(CTX_LABELS[0]))))
        recompute(s);
    ImGui::SameLine();
    ImGui::Text("   Filter:");
    ImGui::SameLine();
    ImGui::InputTextWithHint("##filter", "name / family", s.filter, sizeof(s.filter));

    ImGui::Spacing();
    ImGui::Text("%zu / %zu models fit on this machine.",
                s.ranked.size(), CATALOG_SIZE);
    ImGui::Separator();

    const float em = ImGui::GetFontSize();
    ImGui::BeginChild("left", ImVec2(em * 38, 0), true);
    if (ImGui::BeginTable("ranking", 7,
        ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg
        | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable)) {
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, em * 2.2f);
        ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, em * 3.5f);
        ImGui::TableSetupColumn("Model");
        ImGui::TableSetupColumn("Quant", ImGuiTableColumnFlags_WidthFixed, em * 4.0f);
        ImGui::TableSetupColumn("Ctx", ImGuiTableColumnFlags_WidthFixed, em * 3.5f);
        ImGui::TableSetupColumn("Mem", ImGuiTableColumnFlags_WidthFixed, em * 4.0f);
        ImGui::TableSetupColumn("tok/s", ImGuiTableColumnFlags_WidthFixed, em * 3.8f);
        ImGui::TableHeadersRow();

        std::string filt(s.filter);
        for (size_t i = 0; i < s.ranked.size(); ++i) {
            const auto& m = *s.ranked[i].model;
            // Substring filter
            if (!filt.empty()) {
                std::string n(m.name), id(m.id), fam(m.family);
                std::transform(filt.begin(), filt.end(), filt.begin(),
                                [](unsigned char c){ return std::tolower(c); });
                auto lower = [](std::string x){
                    std::transform(x.begin(), x.end(), x.begin(),
                                    [](unsigned char c){ return std::tolower(c); });
                    return x;
                };
                if (lower(n).find(filt) == std::string::npos
                    && lower(id).find(filt) == std::string::npos
                    && lower(fam).find(filt) == std::string::npos) continue;
            }
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0); ImGui::Text("%zu", i + 1);
            ImGui::TableSetColumnIndex(1);
            float sc = static_cast<float>(s.ranked[i].composite);
            ImVec4 col = sc >= 70 ? ImVec4(0.4f, 1.0f, 0.4f, 1.0f)
                        : sc >= 50 ? ImVec4(1.0f, 1.0f, 0.4f, 1.0f)
                                    : ImVec4(0.9f, 0.9f, 0.9f, 1.0f);
            ImGui::TextColored(col, "%.1f", sc);
            ImGui::TableSetColumnIndex(2);
            std::string name(m.name);
            bool selected = (static_cast<int>(i) == s.selected_model);
            if (ImGui::Selectable(name.c_str(), selected,
                ImGuiSelectableFlags_SpanAllColumns))
                s.selected_model = static_cast<int>(i);
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%s", std::string(s.ranked[i].fit.quant).c_str());
            ImGui::TableSetColumnIndex(4);
            ImGui::Text("%dK", s.ranked[i].fit.context / 1000);
            ImGui::TableSetColumnIndex(5);
            ImGui::Text("%.1fG", to_gb(s.ranked[i].fit.total_bytes));
            ImGui::TableSetColumnIndex(6);
            ImGui::Text("%.1f", s.ranked[i].tokens_per_sec);
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("detail", ImVec2(0, 0), true);
    if (s.selected_model >= 0 && s.selected_model < static_cast<int>(s.ranked.size())) {
        const auto& sm = s.ranked[s.selected_model];
        const auto& m = *sm.model;
        ImGui::TextWrapped("%s", std::string(m.name).c_str());
        ImGui::TextDisabled("%s", std::string(m.id).c_str());
        ImGui::Separator();
        ImGui::Text("Family: %s", std::string(m.family).c_str());
        ImGui::Text("Type:   %s", std::string(m.type).c_str());
        ImGui::Text("Params: %.1fB (active %.1fB)", m.params_b, m.active_params_b);
        ImGui::Text("Max context: %d tokens", m.context_max);
        ImGui::Text("Quality (F16): %.0f / 100", m.quality);
        ImGui::Separator();
        ImGui::Text("Fit");
        if (sm.fit.fits) {
            ImGui::Text("  %s @ %d ctx -> %.1f GB on %s",
                std::string(sm.fit.quant).c_str(), sm.fit.context,
                to_gb(sm.fit.total_bytes), tier_label(sm.fit.tier));
            ImGui::Text("  Budget %.1f GB  ·  ~%.0f tok/s",
                to_gb(sm.fit.budget_bytes), sm.tokens_per_sec);
            ImGui::Separator();
            ImGui::Text("Sub-scores");
            ImGui::ProgressBar(static_cast<float>(sm.quality / 100), ImVec2(0, 0),
                                "Quality");
            ImGui::ProgressBar(static_cast<float>(sm.speed / 100), ImVec2(0, 0),
                                "Speed");
            ImGui::ProgressBar(static_cast<float>(sm.context / 100), ImVec2(0, 0),
                                "Context");
            ImGui::ProgressBar(static_cast<float>(sm.capability / 100), ImVec2(0, 0),
                                "Capability");
            ImGui::Separator();
            ImGui::Text("Composite: %.1f", sm.composite);
        } else {
            ImGui::TextWrapped("Won't fit: %s", sm.fit.reason.c_str());
        }
    } else {
        ImGui::TextDisabled("Select a model on the left to see details.");
    }
    ImGui::EndChild();
}

// -----------------------------------------------------------------------
// Tab: Hardware
// -----------------------------------------------------------------------
static void draw_hardware_tab(AppState& s) {
    const auto& hw = s.hw;
    ImGui::Spacing();
    ImGui::Text("Platform: %s %s", hw.platform.c_str(), hw.arch.c_str());
    ImGui::Separator();
    ImGui::Text("CPU");
    ImGui::Text("  %s", hw.cpu_name.c_str());
    ImGui::Text("  %d cores / %d threads @ %.1f GHz",
                hw.cpu_cores, hw.cpu_threads, hw.cpu_freq_mhz / 1000.0);
    ImGui::Separator();
    ImGui::Text("Memory");
    ImGui::Text("  %.1f GB total  ·  %.1f GB available", to_gb(hw.ram_bytes),
                to_gb(hw.ram_available_bytes));
    ImGui::Text("  Bandwidth: ~%.0f GB/s", hw.ram_bandwidth_gbps);
    ImGui::Separator();
    ImGui::Text("GPU");
    if (hw.gpus.empty()) {
        ImGui::TextDisabled("  none detected — CPU-only inference.");
    } else {
        for (const auto& g : hw.gpus) {
            ImGui::Text("  %s", g.name.c_str());
            ImGui::Text("    %.1f GB VRAM  ·  ~%.0f GB/s  ·  vendor: %s",
                        to_gb(g.vram_bytes), g.bandwidth_gbps, g.vendor.c_str());
        }
    }
    if (hw.unified_memory) {
        ImGui::Separator();
        ImGui::TextDisabled("Apple Silicon: the GPU shares system RAM "
                            "(~75%% available as VRAM).");
    }
}

// -----------------------------------------------------------------------
// Tab: Glossary
// -----------------------------------------------------------------------
static void draw_glossary_tab(AppState& s) {
    char buf[256];
    std::strncpy(buf, s.glossary_search.c_str(), sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    ImGui::Spacing();
    ImGui::Text("Search:");
    ImGui::SameLine();
    if (ImGui::InputTextWithHint("##search", "type to search across all definitions...",
                                   buf, sizeof(buf)))
        s.glossary_search = buf;
    ImGui::Separator();

    // Build filtered/searched list.
    std::vector<const gloss::GlossaryEntry*> shown;
    if (!s.glossary_search.empty()) {
        shown = gloss::search(s.glossary_search);
    } else if (s.selected_category >= 0
                && s.selected_category < static_cast<int>(gloss::CATEGORIES.size())) {
        for (auto term : gloss::CATEGORIES[s.selected_category].terms)
            if (auto e = gloss::lookup(term)) shown.push_back(e);
    } else {
        for (const auto& e : gloss::ENTRIES) shown.push_back(&e);
    }

    // Layout: 3 columns — categories | term list | detail
    const float em_g = ImGui::GetFontSize();
    ImGui::BeginChild("cats", ImVec2(em_g * 12, 0), true);
    ImGui::Text("Categories");
    ImGui::Separator();
    if (ImGui::Selectable("All", s.selected_category == -1)) {
        s.selected_category = -1;
        s.selected_term = -1;
    }
    for (size_t i = 0; i < gloss::CATEGORIES.size(); ++i) {
        const auto& cat = gloss::CATEGORIES[i];
        std::string label(cat.name);
        char count[16]; std::snprintf(count, sizeof(count), " (%zu)", cat.terms.size());
        label += count;
        if (ImGui::Selectable(label.c_str(), s.selected_category == static_cast<int>(i))) {
            s.selected_category = static_cast<int>(i);
            s.selected_term = -1;
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("terms", ImVec2(em_g * 17, 0), true);
    ImGui::Text("Terms (%zu)", shown.size());
    ImGui::Separator();
    for (size_t i = 0; i < shown.size(); ++i) {
        std::string label(shown[i]->term);
        if (ImGui::Selectable(label.c_str(), s.selected_term == static_cast<int>(i))) {
            s.selected_term = static_cast<int>(i);
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("explain", ImVec2(0, 0), true);
    if (s.selected_term >= 0 && s.selected_term < static_cast<int>(shown.size())) {
        const auto& e = *shown[s.selected_term];
        ImGui::TextColored(ImVec4(1, 0.9f, 0.4f, 1), "%s", std::string(e.term).c_str());
        ImGui::TextWrapped("%s", std::string(e.summary).c_str());
        ImGui::Separator();
        ImGui::TextWrapped("%s", std::string(e.body).c_str());
        if (!e.analogy.empty()) {
            ImGui::Separator();
            ImGui::Text("Analogy");
            ImGui::TextWrapped("%s", std::string(e.analogy).c_str());
        }
        if (!e.see_also.empty()) {
            ImGui::Separator();
            std::string sa = "See also: ";
            for (size_t i = 0; i < e.see_also.size(); ++i) {
                sa += std::string(e.see_also[i]);
                if (i + 1 < e.see_also.size()) sa += ", ";
            }
            ImGui::TextDisabled("%s", sa.c_str());
        }
    } else {
        ImGui::TextDisabled("Select a term to read about it.");
    }
    ImGui::EndChild();
}

// -----------------------------------------------------------------------
// Main loop
// -----------------------------------------------------------------------
int main() {
    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) {
        std::fprintf(stderr, "GLFW init failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);
    // NOTE: GLFW_SCALE_TO_MONITOR makes the window 2× on HiDPI platforms,
    // which we don't want — our HiDPI math operates on logical sizes already.

    // Clamp the initial window to the primary monitor's work area so it
    // never opens larger than the screen, regardless of zoom.
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    int wx = 0, wy = 0, ww = 1920, wh = 1080;
    if (monitor) {
        glfwGetMonitorWorkarea(monitor, &wx, &wy, &ww, &wh);
    }
    const int desired_w = 1280;
    const int desired_h = 800;
    int win_w = std::min(desired_w, std::max(800, ww - 80));
    int win_h = std::min(desired_h, std::max(600, wh - 120));

    GLFWwindow* window = glfwCreateWindow(win_w, win_h, "ModelFit", nullptr, nullptr);
    if (!window) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;  // don't litter the cwd with imgui.ini

    ZoomState zoom;
    setup_initial_style(zoom, window);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    AppState state;
    state.hw = detect_hardware();
    recompute(state);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Rebuild the font atlas *before* NewFrame if the user changed zoom.
        if (zoom.fonts_dirty) {
            ImGui_ImplOpenGL3_DestroyFontsTexture();
            rebuild_fonts(zoom);
            ImGui_ImplOpenGL3_CreateFontsTexture();
            zoom.fonts_dirty = false;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // -------- Keyboard shortcuts (Cmd on macOS, Ctrl elsewhere) --------
        ImGuiIO& nio = ImGui::GetIO();
        bool mod =
#if defined(__APPLE__)
            nio.KeySuper;
#else
            nio.KeyCtrl;
#endif
        if (mod) {
            if (ImGui::IsKeyPressed(ImGuiKey_Equal, false)
                || ImGui::IsKeyPressed(ImGuiKey_KeypadAdd, false))
                zoom_in(zoom);
            else if (ImGui::IsKeyPressed(ImGuiKey_Minus, false)
                       || ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract, false))
                zoom_out(zoom);
            else if (ImGui::IsKeyPressed(ImGuiKey_0, false))
                zoom_reset(zoom);
        }
        // Ctrl + mouse wheel — common zoom gesture.
        if (mod && nio.MouseWheel != 0.0f) {
            if (nio.MouseWheel > 0) zoom_in(zoom); else zoom_out(zoom);
        }

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar
                                | ImGuiWindowFlags_NoResize
                                | ImGuiWindowFlags_NoMove
                                | ImGuiWindowFlags_NoCollapse
                                | ImGuiWindowFlags_NoBringToFrontOnFocus;
        ImGui::Begin("ModelFit", nullptr, flags);
        ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f),
                            "ModelFit — %zu open-weight LLMs ranked for your machine",
                            CATALOG_SIZE);

        // -------- Toolbar: zoom widget + refresh, right-aligned --------
        const float em = ImGui::GetFontSize();
        const float toolbar_w = em * 22.0f;  // approx width of all the buttons + label
        ImGui::SameLine(ImGui::GetWindowWidth() - toolbar_w);
        ImGui::PushButtonRepeat(true);
        if (ImGui::SmallButton("-")) zoom_out(zoom);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Zoom out  (%s -)", mod ? "Cmd" : "Ctrl");
        ImGui::SameLine();
        char zbuf[16]; std::snprintf(zbuf, sizeof(zbuf), "%.0f%%", zoom.user_zoom * 100);
        if (ImGui::SmallButton(zbuf)) zoom_reset(zoom);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Reset zoom to 100%%  (%s 0)", mod ? "Cmd" : "Ctrl");
        ImGui::SameLine();
        if (ImGui::SmallButton("+")) zoom_in(zoom);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Zoom in  (%s +)", mod ? "Cmd" : "Ctrl");
        ImGui::PopButtonRepeat();
        ImGui::SameLine();
        if (ImGui::SmallButton("Refresh hardware")) {
            state.hw = detect_hardware();
            recompute(state);
        }

        ImGui::Spacing();
        if (ImGui::BeginTabBar("tabs")) {
            if (ImGui::BeginTabItem("Models")) { draw_models_tab(state); ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("Hardware")) { draw_hardware_tab(state); ImGui::EndTabItem(); }
            if (ImGui::BeginTabItem("Glossary")) { draw_glossary_tab(state); ImGui::EndTabItem(); }
            ImGui::EndTabBar();
        }
        ImGui::End();

        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.06f, 0.06f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
