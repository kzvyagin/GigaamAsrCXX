#pragma once

#include <filesystem>
#include <string>

#include "app/Ports.hpp"

namespace gigaam::infra::download {

class HttplibHttpClient : public app::IHttpClient {
public:
    std::string GetText(const std::string &url) const override;
    void DownloadToFile(const std::string &url, const std::filesystem::path &output_file) const override;
};

}  // namespace gigaam::infra::download
