#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace gigaam::domain {

struct DatasetSample {
    std::string audio_url;
    std::filesystem::path relative_audio_path;
    std::string reference_text;
};

std::filesystem::path DefaultDatasetDirectory();
std::vector<DatasetSample> TrimDatasetSamples(const std::vector<DatasetSample> &samples, size_t limit);

}  // namespace gigaam::domain
