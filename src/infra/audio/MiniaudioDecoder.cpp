#define MA_ENABLE_VORBIS
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include "infra/audio/MiniaudioDecoder.hpp"

#include <stdexcept>
#include <vector>

namespace gigaam::infra::audio {

domain::AudioBuffer MiniaudioDecoder::DecodeFile(const std::filesystem::path &audio_file) const {
    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 0, 0);

    ma_decoder decoder;
    const ma_result init_result = ma_decoder_init_file(audio_file.string().c_str(), &config, &decoder);
    if (init_result != MA_SUCCESS) {
        throw std::runtime_error("Cannot decode audio file: " + audio_file.string());
    }

    ma_uint64 frame_count = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count) != MA_SUCCESS || frame_count == 0) {
        ma_decoder_uninit(&decoder);
        throw std::runtime_error("Audio file contains no PCM frames: " + audio_file.string());
    }

    const ma_uint32 channels = decoder.outputChannels;
    if (channels == 0) {
        ma_decoder_uninit(&decoder);
        throw std::runtime_error("Decoded audio reports zero channels: " + audio_file.string());
    }
    const int sample_rate = static_cast<int>(decoder.outputSampleRate);
    std::vector<float> interleaved(static_cast<size_t>(frame_count * channels));
    ma_uint64 read_frames = 0;
    const ma_result read_result = ma_decoder_read_pcm_frames(&decoder, interleaved.data(), frame_count, &read_frames);
    ma_decoder_uninit(&decoder);

    if (read_result != MA_SUCCESS || read_frames == 0) {
        throw std::runtime_error("Failed to read decoded audio frames: " + audio_file.string());
    }

    std::vector<float> mono(static_cast<size_t>(read_frames));
    for (ma_uint64 frame = 0; frame < read_frames; ++frame) {
        float mixed = 0.0f;
        for (ma_uint32 channel = 0; channel < channels; ++channel) {
            mixed += interleaved[static_cast<size_t>(frame * channels + channel)];
        }
        mono[static_cast<size_t>(frame)] = mixed / static_cast<float>(channels);
    }

    return {sample_rate, std::move(mono)};
}

}  // namespace gigaam::infra::audio
