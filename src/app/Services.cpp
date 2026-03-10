#include "app/Services.hpp"

#include <filesystem>
#include <stdexcept>

#include "domain/Assets.hpp"
#include "domain/Dataset.hpp"

namespace gigaam::app {

InferenceService::InferenceService(const IAudioDecoder &decoder, const IRecognizer &recognizer)
    : decoder_(decoder), recognizer_(recognizer) {}

std::string InferenceService::Run(const std::filesystem::path &audio_file) const {
    const auto audio = decoder_.DecodeFile(audio_file);
    return recognizer_.Recognize(audio);
}

EvaluationService::EvaluationService(const InferenceService &inference)
    : inference_(inference) {}

EvalReport EvaluationService::Run(const std::filesystem::path &manifest_file) const {
    const auto entries = domain::LoadManifest(manifest_file);

    EvalReport report;
    report.items.reserve(entries.size());
    for (const auto &entry : entries) {
        const auto hypothesis = inference_.Run(entry.audio_path);
        domain::UpdateTotals(report.totals, entry.reference_text, hypothesis);
        report.items.push_back({entry.audio_path, entry.reference_text, hypothesis});
    }

    return report;
}

ModelDownloadService::ModelDownloadService(const IHttpClient &http_client)
    : http_client_(http_client) {}

void ModelDownloadService::Run(const std::filesystem::path &output_dir, bool force, std::ostream &out) const {
    std::filesystem::create_directories(output_dir);

    for (const auto &asset : domain::DefaultE2eModelAssets()) {
        const auto output_file = output_dir / asset.file_name;
        if (!force && std::filesystem::exists(output_file)) {
            out << "Skipping existing model asset: " << output_file << '\n';
            continue;
        }

        out << "Downloading " << asset.file_name << '\n';
        http_client_.DownloadToFile(asset.url, output_file);
    }
}

DatasetDownloadService::DatasetDownloadService(const IHttpClient &http_client, const IDatasetIndex &dataset_index)
    : http_client_(http_client), dataset_index_(dataset_index) {}

void DatasetDownloadService::Run(const std::filesystem::path &output_dir, size_t limit, bool force, std::ostream &out) const {
    const auto samples = dataset_index_.ListSamples(limit);
    if (samples.empty()) {
        throw std::runtime_error("Dataset index returned zero samples.");
    }

    const auto audio_dir = output_dir / "audio";
    std::filesystem::create_directories(audio_dir);

    std::vector<domain::ManifestEntry> manifest_entries;
    manifest_entries.reserve(samples.size());

    for (const auto &sample : samples) {
        const auto output_file = output_dir / sample.relative_audio_path;
        std::filesystem::create_directories(output_file.parent_path());

        if (!force && std::filesystem::exists(output_file)) {
            out << "Skipping existing dataset file: " << output_file << '\n';
        } else {
            out << "Downloading " << sample.relative_audio_path.generic_string() << '\n';
            http_client_.DownloadToFile(sample.audio_url, output_file);
        }

        manifest_entries.push_back({sample.relative_audio_path, sample.reference_text});
    }

    domain::WriteManifest(output_dir / "manifest.tsv", manifest_entries);
}

}  // namespace gigaam::app
