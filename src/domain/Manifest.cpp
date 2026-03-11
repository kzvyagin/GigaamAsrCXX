#include "domain/Manifest.hpp"

#include <fstream>
#include <stdexcept>

namespace gigaam::domain {

namespace {

std::string TrimAsciiWhitespace(const std::string &value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

}  // namespace

std::vector<ManifestEntry> LoadManifest(const std::filesystem::path &manifest_file) {
    std::ifstream input(manifest_file);
    if (!input) {
        throw std::runtime_error("Cannot open manifest file: " + manifest_file.string());
    }

    const std::filesystem::path base_dir = manifest_file.parent_path();
    std::vector<ManifestEntry> entries;
    std::string line;
    size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        const size_t tab_pos = line.find('\t');
        if (tab_pos == std::string::npos) {
            throw std::runtime_error(
                "Manifest line must contain a TAB separator at line " + std::to_string(line_number));
        }

        std::filesystem::path audio_path = line.substr(0, tab_pos);
        if (audio_path.is_relative()) {
            audio_path = base_dir / audio_path;
        }

        entries.push_back({audio_path, TrimAsciiWhitespace(line.substr(tab_pos + 1))});
    }

    if (entries.empty()) {
        throw std::runtime_error("Manifest is empty: " + manifest_file.string());
    }

    return entries;
}

void WriteManifest(const std::filesystem::path &manifest_file, const std::vector<ManifestEntry> &entries) {
    if (entries.empty()) {
        throw std::runtime_error("Cannot write an empty manifest.");
    }

    std::filesystem::create_directories(manifest_file.parent_path());
    std::ofstream output(manifest_file);
    if (!output) {
        throw std::runtime_error("Cannot open manifest for writing: " + manifest_file.string());
    }

    for (const auto &entry : entries) {
        output << entry.audio_path.generic_string() << '\t' << entry.reference_text << '\n';
    }
}

}  // namespace gigaam::domain
