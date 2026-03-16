#pragma once

#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

#include "app/Ports.hpp"
#include "domain/Manifest.hpp"
#include "domain/Text.hpp"

namespace gigaam::app {

struct EvalItemResult {
    std::filesystem::path audio_path;
    std::string reference_text;
    std::string hypothesis_text;
};

struct EvalReport {
    gigaam::domain::EvalTotals totals;
    std::vector<EvalItemResult> items;
};

class InferenceService {
public:
    InferenceService(const IAudioDecoder &decoder, const IRecognizer &recognizer);

    std::string Run(const std::filesystem::path &audio_file) const;

private:
    const IAudioDecoder &decoder_;
    const IRecognizer &recognizer_;
};

class EvaluationService {
public:
    explicit EvaluationService(const InferenceService &inference);

    EvalReport Run(const std::filesystem::path &manifest_file) const;

private:
    const InferenceService &inference_;
};

class ModelDownloadService {
public:
    explicit ModelDownloadService(const IHttpClient &http_client);

    void Run(const std::filesystem::path &output_dir, bool force, std::ostream &out) const;

private:
    const IHttpClient &http_client_;
};

class DatasetDownloadService {
public:
    DatasetDownloadService(const IHttpClient &http_client, const IDatasetIndex &dataset_index);

    void Run(const std::filesystem::path &output_dir, size_t limit, bool force, std::ostream &out) const;

private:
    const IHttpClient &http_client_;
    const IDatasetIndex &dataset_index_;
};

}  // namespace gigaam::app
