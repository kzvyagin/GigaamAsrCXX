#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace gigaam::domain {

struct ModelAsset {
    std::string file_name;
    std::string url;
};

struct ModelLayout {
    std::filesystem::path model_dir;
    std::filesystem::path encoder;
    std::filesystem::path decoder;
    std::filesystem::path joint;
    std::filesystem::path vocab;
};

std::vector<ModelAsset> DefaultE2eModelAssets();
ModelLayout ResolveModelLayout(const std::filesystem::path &model_dir);
std::filesystem::path DefaultModelDirectory();

}  // namespace gigaam::domain
