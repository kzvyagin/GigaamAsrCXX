#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "domain/Audio.hpp"
#include "domain/Dataset.hpp"

namespace gigaam::app {

class IAudioDecoder {
public:
    virtual ~IAudioDecoder() = default;
    virtual domain::AudioBuffer DecodeFile(const std::filesystem::path &audio_file) const = 0;
};

class IRecognizer {
public:
    virtual ~IRecognizer() = default;
    virtual std::string Recognize(const domain::AudioBuffer &audio) const = 0;
};

class IHttpClient {
public:
    virtual ~IHttpClient() = default;
    virtual std::string GetText(const std::string &url) const = 0;
    virtual void DownloadToFile(const std::string &url, const std::filesystem::path &output_file) const = 0;
};

class IDatasetIndex {
public:
    virtual ~IDatasetIndex() = default;
    virtual std::vector<domain::DatasetSample> ListSamples(size_t limit) const = 0;
};

}  // namespace gigaam::app
