#include "domain/Dataset.hpp"

#include <algorithm>
#include <stdexcept>

namespace gigaam::domain {

std::filesystem::path DefaultDatasetDirectory() {
    return std::filesystem::path("data") / "ru-public";
}

std::vector<DatasetSample> TrimDatasetSamples(const std::vector<DatasetSample> &samples, size_t limit) {
    if (limit == 0) {
        throw std::runtime_error("Dataset sample limit must be greater than zero.");
    }

    const size_t actual = std::min(samples.size(), limit);
    return {samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(actual)};
}

}  // namespace gigaam::domain
