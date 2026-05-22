#include "modelfit/glossary.h"

#include <algorithm>
#include <cctype>
#include <vector>

namespace modelfit {
namespace gloss {

static std::string lower(std::string_view sv) {
    std::string s(sv);
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c){ return std::tolower(c); });
    return s;
}

static bool contains(std::string_view hay, std::string_view needle) {
    return hay.find(needle) != std::string_view::npos;
}

const GlossaryEntry* lookup(std::string_view term) {
    auto q = lower(term);
    for (const auto& e : ENTRIES)
        if (lower(e.term) == q) return &e;
    return nullptr;
}

std::vector<const GlossaryEntry*> search(std::string_view query, std::size_t limit) {
    auto q = lower(query);
    if (q.empty()) return {};
    // Tokenize on whitespace, hyphens, underscores.
    std::vector<std::string> tokens;
    std::string cur;
    for (char c : q) {
        if (c == ' ' || c == '-' || c == '_') {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
        } else cur += c;
    }
    if (!cur.empty()) tokens.push_back(cur);
    if (tokens.empty()) return {};

    std::vector<std::pair<int, const GlossaryEntry*>> scored;
    for (const auto& e : ENTRIES) {
        auto term_l = lower(e.term);
        auto sum_l = lower(e.summary);
        auto body_l = lower(e.body);
        auto ana_l = lower(e.analogy);
        std::string all = term_l + " " + sum_l + " " + body_l + " " + ana_l;
        // All tokens must appear somewhere.
        bool all_match = true;
        for (const auto& t : tokens)
            if (!contains(all, t)) { all_match = false; break; }
        if (!all_match) continue;

        int score = 0;
        if (term_l == q) score += 200;
        if (contains(term_l, q)) score += 80;
        for (const auto& t : tokens) {
            if (contains(term_l, t)) score += 30;
            if (contains(sum_l, t))  score += 15;
            if (contains(body_l, t)) score += 5;
            if (contains(ana_l, t))  score += 3;
        }
        scored.emplace_back(score, &e);
    }
    std::sort(scored.begin(), scored.end(),
                [](const auto& a, const auto& b){
                    if (a.first != b.first) return a.first > b.first;
                    return std::string(a.second->term) < std::string(b.second->term);
                });
    std::vector<const GlossaryEntry*> out;
    out.reserve(std::min(limit, scored.size()));
    for (size_t i = 0; i < scored.size() && i < limit; ++i)
        out.push_back(scored[i].second);
    return out;
}

} // namespace gloss
} // namespace modelfit
