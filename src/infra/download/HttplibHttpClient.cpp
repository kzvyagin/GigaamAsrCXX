#include "infra/download/HttplibHttpClient.hpp"

#include <cstdio>
#include <fstream>
#include <regex>
#include <stdexcept>

#include <httplib.h>

namespace gigaam::infra::download {

namespace {

struct ParsedUrl {
    std::string scheme;
    std::string host;
    std::string path_and_query;
};

ParsedUrl ParseUrl(const std::string &url) {
    static const std::regex pattern(R"(^(https?)://([^/]+)(/.*)$)");
    std::smatch match;
    if (!std::regex_match(url, match, pattern)) {
        throw std::runtime_error("Unsupported URL: " + url);
    }

    return {match[1].str(), match[2].str(), match[3].str()};
}

std::string ResolveRedirect(const std::string &current_url, const std::string &location) {
    if (location.rfind("http://", 0) == 0 || location.rfind("https://", 0) == 0) {
        return location;
    }

    const auto current = ParseUrl(current_url);
    if (!location.empty() && location.front() == '/') {
        return current.scheme + "://" + current.host + location;
    }

    const auto slash = current.path_and_query.find_last_of('/');
    const std::string base_path = slash == std::string::npos ? "/" : current.path_and_query.substr(0, slash + 1);
    return current.scheme + "://" + current.host + base_path + location;
}

std::string ShellEscape(const std::string &value) {
    std::string escaped = "'";
    for (char ch : value) {
        if (ch == '\'') {
            escaped += "'\\''";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped += "'";
    return escaped;
}

httplib::Result PerformGet(const std::string &url,
                           const httplib::ContentReceiver &receiver,
                           int redirects_left) {
    if (redirects_left < 0) {
        throw std::runtime_error("Too many HTTP redirects while downloading: " + url);
    }

    const auto parsed = ParseUrl(url);
    httplib::Result result;
    if (parsed.scheme == "https") {
        httplib::SSLClient client(parsed.host);
        client.enable_server_certificate_verification(true);
        client.set_follow_location(false);
        client.set_connection_timeout(30, 0);
        client.set_read_timeout(300, 0);
        client.set_write_timeout(300, 0);
        result = client.Get(parsed.path_and_query.c_str(), receiver);
    } else {
        httplib::Client client(parsed.host);
        client.set_follow_location(false);
        client.set_connection_timeout(30, 0);
        client.set_read_timeout(300, 0);
        client.set_write_timeout(300, 0);
        result = client.Get(parsed.path_and_query.c_str(), receiver);
    }
    if (!result) {
        throw std::runtime_error("HTTP GET failed for: " + url);
    }

    if (result->status >= 300 && result->status < 400) {
        const auto location = result->get_header_value("Location");
        if (location.empty()) {
            throw std::runtime_error("Redirect response without Location header: " + url);
        }
        return PerformGet(ResolveRedirect(url, location), receiver, redirects_left - 1);
    }

    if (result->status < 200 || result->status >= 300) {
        throw std::runtime_error("Unexpected HTTP status " + std::to_string(result->status) + " for " + url);
    }

    return result;
}

std::string GetTextWithCurl(const std::string &url) {
    const std::string command = "curl -fsSL " + ShellEscape(url);
    FILE *pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("Failed to execute curl for: " + url);
    }

    std::string output;
    char buffer[4096];
    while (const size_t read = std::fread(buffer, 1, sizeof(buffer), pipe)) {
        output.append(buffer, read);
    }

    if (pclose(pipe) != 0) {
        throw std::runtime_error("curl failed for: " + url);
    }
    return output;
}

void DownloadWithCurl(const std::string &url, const std::filesystem::path &output_file) {
    std::filesystem::create_directories(output_file.parent_path());
    const std::string command = "curl -fsSL " + ShellEscape(url) + " -o " + ShellEscape(output_file.string());
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("curl failed for: " + url);
    }
}

}  // namespace

std::string HttplibHttpClient::GetText(const std::string &url) const {
    std::string body;
    try {
        PerformGet(url, [&](const char *data, size_t length) {
            body.append(data, length);
            return true;
        }, 5);
        return body;
    } catch (const std::exception &) {
        return GetTextWithCurl(url);
    }
}

void HttplibHttpClient::DownloadToFile(const std::string &url, const std::filesystem::path &output_file) const {
    try {
        std::filesystem::create_directories(output_file.parent_path());
        std::ofstream output(output_file, std::ios::binary);
        if (!output) {
            throw std::runtime_error("Cannot open output file for download: " + output_file.string());
        }

        PerformGet(url, [&](const char *data, size_t length) {
            output.write(data, static_cast<std::streamsize>(length));
            return static_cast<bool>(output);
        }, 5);
    } catch (const std::exception &) {
        DownloadWithCurl(url, output_file);
    }
}

}  // namespace gigaam::infra::download
