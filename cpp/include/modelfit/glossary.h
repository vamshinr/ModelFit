// Glossary structure shared by GUI search and CLI explainer.
#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace modelfit {
namespace gloss {

struct GlossaryEntry {
    std::string_view term;
    std::string_view summary;
    std::string_view body;
    std::string_view analogy;
    std::vector<std::string_view> see_also;
};

struct Category {
    std::string_view name;
    std::vector<std::string_view> terms;
};

// Generated arrays — defined in src/glossary_data.cpp by gen_glossary.py.
extern const std::array<GlossaryEntry, 109> ENTRIES;
extern const std::array<Category, 10> CATEGORIES;

// Token-based search across term/summary/body/analogy.
std::vector<const GlossaryEntry*> search(std::string_view query, std::size_t limit = 25);

const GlossaryEntry* lookup(std::string_view term);

} // namespace gloss
} // namespace modelfit
