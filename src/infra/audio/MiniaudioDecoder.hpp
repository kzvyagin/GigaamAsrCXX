#pragma once

#include "app/Ports.hpp"

namespace gigaam::infra::audio {

class MiniaudioDecoder : public app::IAudioDecoder {
public:
    domain::AudioBuffer DecodeFile(const std::filesystem::path &audio_file) const override;
};

}  // namespace gigaam::infra::audio
