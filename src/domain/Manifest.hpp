#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace gigaam::domain {

struct ManifestEntry {
    std::filesystem::path audio_path;
    std::string reference_text;
};

std::vector<ManifestEntry> LoadManifest(const std::filesystem::path &manifest_file);
void WriteManifest(const std::filesystem::path &manifest_file, const std::vector<ManifestEntry> &entries);

}  // namespace gigaam::domain
