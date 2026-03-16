#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gigaam::domain {

struct EvalTotals {
    size_t utterances = 0;
    size_t word_edits = 0;
    size_t word_total = 0;
    size_t char_edits = 0;
    size_t char_total = 0;
};

std::vector<uint32_t> Utf8ToCodepoints(const std::string &text);
std::string CodepointsToUtf8(const std::vector<uint32_t> &codepoints);
std::string NormalizeTextForMetrics(const std::string &text);
std::vector<std::string> SplitWords(const std::string &text);
void UpdateTotals(EvalTotals &totals, const std::string &reference, const std::string &hypothesis);
double ComputeWerPercent(const EvalTotals &totals);
double ComputeCerPercent(const EvalTotals &totals);

}  // namespace gigaam::domain
