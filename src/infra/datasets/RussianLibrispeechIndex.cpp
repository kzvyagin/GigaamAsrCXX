#include "infra/datasets/RussianLibrispeechIndex.hpp"

#include <stdexcept>

#include <nlohmann/json.hpp>

#include "domain/Dataset.hpp"

namespace gigaam::infra::datasets {

namespace {

constexpr char kDatasetUrl[] =
    "https://datasets-server.huggingface.co/first-rows"
    "?dataset=istupakov%2Frussian_librispeech&config=default&split=test";

}  // namespace

RussianLibrispeechIndex::RussianLibrispeechIndex(const app::IHttpClient &http_client)
    : http_client_(http_client) {}

std::vector<domain::DatasetSample> RussianLibrispeechIndex::ListSamples(size_t limit) const {
    const auto payload = http_client_.GetText(kDatasetUrl);
    const auto json = nlohmann::json::parse(payload);

    const auto rows_it = json.find("rows");
    if (rows_it == json.end() || !rows_it->is_array()) {
        throw std::runtime_error("Dataset response does not contain rows array.");
    }

    std::vector<domain::DatasetSample> samples;
    samples.reserve(rows_it->size());

    size_t index = 0;
    for (const auto &item : *rows_it) {
        if (index >= limit) {
            break;
        }

        const auto &row = item.at("row");
        const auto &audio_items = row.at("audio");
        if (!audio_items.is_array() || audio_items.empty()) {
            throw std::runtime_error("Dataset row does not contain audio sources.");
        }

        ++index;
        const std::string file_name = "sample_" + std::string(index < 10 ? "00" : index < 100 ? "0" : "") +
                                      std::to_string(index) + ".wav";
        samples.push_back({
            audio_items.at(0).at("src").get<std::string>(),
            std::filesystem::path("audio") / file_name,
            row.at("text").get<std::string>(),
        });
    }

    return domain::TrimDatasetSamples(samples, limit);
}

}  // namespace gigaam::infra::datasets
