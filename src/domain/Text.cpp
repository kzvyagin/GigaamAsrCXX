#include "domain/Text.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace gigaam::domain {

namespace {

uint32_t NormalizeRussianCodepoint(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') {
        return cp + 32;
    }
    if (cp == 0x0401 || cp == 0x0451) {
        return 0x0435;
    }
    if (cp >= 0x0410 && cp <= 0x042F) {
        return cp + 32;
    }
    return cp;
}

bool IsNormalizedWordCodepoint(uint32_t cp) {
    return (cp >= 'a' && cp <= 'z') ||
           (cp >= '0' && cp <= '9') ||
           (cp >= 0x0430 && cp <= 0x044F);
}

template <typename T>
size_t EditDistance(const std::vector<T> &a, const std::vector<T> &b) {
    std::vector<size_t> prev(b.size() + 1);
    std::vector<size_t> curr(b.size() + 1);

    for (size_t j = 0; j <= b.size(); ++j) {
        prev[j] = j;
    }

    for (size_t i = 1; i <= a.size(); ++i) {
        curr[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            const size_t cost = a[i - 1] == b[j - 1] ? 0 : 1;
            curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
        }
        std::swap(prev, curr);
    }

    return prev[b.size()];
}

}  // namespace

std::vector<std::string> SplitWords(const std::string &text) {
    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

std::vector<uint32_t> Utf8ToCodepoints(const std::string &text) {
    std::vector<uint32_t> result;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if ((c & 0x80) == 0) {
            result.push_back(c);
            ++i;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            result.push_back(((c & 0x1F) << 6) |
                             (static_cast<unsigned char>(text[i + 1]) & 0x3F));
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            result.push_back(((c & 0x0F) << 12) |
                             ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                             (static_cast<unsigned char>(text[i + 2]) & 0x3F));
            i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
            result.push_back(((c & 0x07) << 18) |
                             ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
                             ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
                             (static_cast<unsigned char>(text[i + 3]) & 0x3F));
            i += 4;
        } else {
            throw std::runtime_error("Invalid UTF-8 text.");
        }
    }
    return result;
}

std::string CodepointsToUtf8(const std::vector<uint32_t> &codepoints) {
    std::string text;
    for (uint32_t cp : codepoints) {
        if (cp <= 0x7F) {
            text.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            text.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            text.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            text.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            text.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            text.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            text.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            text.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            text.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            text.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return text;
}

std::string NormalizeTextForMetrics(const std::string &text) {
    const auto codepoints = Utf8ToCodepoints(text);

    std::vector<uint32_t> normalized;
    normalized.reserve(codepoints.size());

    bool pending_space = false;
    for (uint32_t cp : codepoints) {
        cp = NormalizeRussianCodepoint(cp);
        if (IsNormalizedWordCodepoint(cp)) {
            if (pending_space && !normalized.empty()) {
                normalized.push_back(' ');
            }
            normalized.push_back(cp);
            pending_space = false;
        } else {
            pending_space = true;
        }
    }

    return CodepointsToUtf8(normalized);
}

void UpdateTotals(EvalTotals &totals, const std::string &reference, const std::string &hypothesis) {
    const std::string normalized_reference = NormalizeTextForMetrics(reference);
    const std::string normalized_hypothesis = NormalizeTextForMetrics(hypothesis);

    const auto ref_words = SplitWords(normalized_reference);
    const auto hyp_words = SplitWords(normalized_hypothesis);
    const auto ref_chars = Utf8ToCodepoints(normalized_reference);
    const auto hyp_chars = Utf8ToCodepoints(normalized_hypothesis);

    totals.utterances += 1;
    totals.word_edits += EditDistance(ref_words, hyp_words);
    totals.word_total += ref_words.size();
    totals.char_edits += EditDistance(ref_chars, hyp_chars);
    totals.char_total += ref_chars.size();
}

double ComputeWerPercent(const EvalTotals &totals) {
    return totals.word_total == 0 ? 0.0 : 100.0 * static_cast<double>(totals.word_edits) / totals.word_total;
}

double ComputeCerPercent(const EvalTotals &totals) {
    return totals.char_total == 0 ? 0.0 : 100.0 * static_cast<double>(totals.char_edits) / totals.char_total;
}

}  // namespace gigaam::domain
