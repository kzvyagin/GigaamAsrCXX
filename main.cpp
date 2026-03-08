#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace {

constexpr int kTargetSampleRate = 16000;
constexpr int kFeatureBins = 64;
constexpr int kFftSize = 320;
constexpr int kHopSize = 160;
constexpr int kMaxTokensPerStep = 3;
constexpr float kPi = 3.14159265358979323846f;

struct WavData {
    int sample_rate = 0;
    std::vector<float> samples;
};

struct TensorNameCache {
    std::vector<std::string> storage;
    std::vector<const char *> ptrs;
};

struct EncoderOutput {
    std::vector<float> frames;
    int64_t time_steps = 0;
    int64_t feature_dim = 0;
};

struct DecoderState {
    std::vector<float> h;
    std::vector<float> c;
};

uint16_t ReadLe16(const uint8_t *p) {
    return static_cast<uint16_t>(p[0]) |
           (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t ReadLe32(const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

int32_t ReadLe24Signed(const uint8_t *p) {
    int32_t value = static_cast<int32_t>(p[0]) |
                    (static_cast<int32_t>(p[1]) << 8) |
                    (static_cast<int32_t>(p[2]) << 16);
    if ((value & 0x00800000) != 0) {
        value |= ~0x00FFFFFF;
    }
    return value;
}

float DecodePcmSample(const uint8_t *p, uint16_t audio_format, uint16_t bits_per_sample) {
    if (audio_format == 3 && bits_per_sample == 32) {
        float value = 0.0f;
        std::memcpy(&value, p, sizeof(value));
        return value;
    }

    if (audio_format != 1) {
        throw std::runtime_error("Only PCM and IEEE float WAV files are supported.");
    }

    switch (bits_per_sample) {
        case 8:
            return static_cast<float>(static_cast<int>(p[0]) - 128) / 128.0f;
        case 16:
            return static_cast<float>(static_cast<int16_t>(ReadLe16(p))) / 32768.0f;
        case 24:
            return static_cast<float>(ReadLe24Signed(p)) / 8388608.0f;
        case 32:
            return static_cast<float>(static_cast<int32_t>(ReadLe32(p))) / 2147483648.0f;
        default:
            throw std::runtime_error("Unsupported PCM bit depth in WAV file.");
    }
}

WavData ReadWavFile(const std::filesystem::path &filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open WAV file: " + filename.string());
    }

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
    if (bytes.size() < 44) {
        throw std::runtime_error("WAV file is too small: " + filename.string());
    }
    if (std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("Unsupported WAV container (expected RIFF/WAVE).");
    }

    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint16_t bits_per_sample = 0;
    uint32_t sample_rate = 0;
    uint16_t block_align = 0;
    const uint8_t *data_ptr = nullptr;
    uint32_t data_size = 0;

    size_t offset = 12;
    while (offset + 8 <= bytes.size()) {
        const char *chunk_id = reinterpret_cast<const char *>(bytes.data() + offset);
        const uint32_t chunk_size = ReadLe32(bytes.data() + offset + 4);
        offset += 8;

        if (offset + chunk_size > bytes.size()) {
            throw std::runtime_error("Corrupted WAV file: chunk exceeds file size.");
        }

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_size < 16) {
                throw std::runtime_error("Invalid WAV fmt chunk.");
            }
            const uint8_t *fmt = bytes.data() + offset;
            audio_format = ReadLe16(fmt + 0);
            num_channels = ReadLe16(fmt + 2);
            sample_rate = ReadLe32(fmt + 4);
            block_align = ReadLe16(fmt + 12);
            bits_per_sample = ReadLe16(fmt + 14);
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            data_ptr = bytes.data() + offset;
            data_size = chunk_size;
        }

        offset += chunk_size + (chunk_size % 2);
    }

    if (audio_format == 0 || num_channels == 0 || sample_rate == 0 || block_align == 0 || data_ptr == nullptr) {
        throw std::runtime_error("Incomplete WAV file: fmt/data chunks are missing.");
    }

    const size_t bytes_per_sample = bits_per_sample / 8;
    if (bytes_per_sample == 0 || block_align != num_channels * bytes_per_sample) {
        throw std::runtime_error("Unsupported WAV block alignment.");
    }

    const size_t total_frames = data_size / block_align;
    if (total_frames == 0) {
        throw std::runtime_error("WAV file contains no audio frames.");
    }

    WavData wav;
    wav.sample_rate = static_cast<int>(sample_rate);
    wav.samples.resize(total_frames);

    for (size_t frame = 0; frame < total_frames; ++frame) {
        float mixed = 0.0f;
        const uint8_t *frame_ptr = data_ptr + frame * block_align;
        for (uint16_t channel = 0; channel < num_channels; ++channel) {
            mixed += DecodePcmSample(frame_ptr + channel * bytes_per_sample, audio_format, bits_per_sample);
        }
        wav.samples[frame] = mixed / static_cast<float>(num_channels);
    }

    return wav;
}

std::vector<float> ResampleLinear(const std::vector<float> &input, int input_sample_rate, int output_sample_rate) {
    if (input.empty()) {
        return {};
    }
    if (input_sample_rate == output_sample_rate) {
        return input;
    }

    const double ratio = static_cast<double>(output_sample_rate) / input_sample_rate;
    const size_t output_size = static_cast<size_t>(input.size() * ratio);
    std::vector<float> output(output_size);

    for (size_t i = 0; i < output_size; ++i) {
        const double src_index = static_cast<double>(i) / ratio;
        const size_t left = static_cast<size_t>(src_index);
        const size_t right = std::min(left + 1, input.size() - 1);
        const double alpha = src_index - left;
        output[i] = static_cast<float>((1.0 - alpha) * input[left] + alpha * input[right]);
    }

    return output;
}

TensorNameCache CollectNames(Ort::Session &session, bool inputs) {
    TensorNameCache cache;
    const size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
    cache.storage.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        auto name = inputs
                        ? session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions())
                        : session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        cache.storage.emplace_back(name.get());
    }

    cache.ptrs.reserve(cache.storage.size());
    for (const auto &name : cache.storage) {
        cache.ptrs.push_back(name.c_str());
    }

    return cache;
}

std::vector<std::string> LoadTokens(const std::filesystem::path &filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open tokens file: " + filename.string());
    }

    std::vector<std::string> id_to_token;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        const size_t split = line.find_last_of(" \t");
        if (split == std::string::npos || split + 1 >= line.size()) {
            throw std::runtime_error("Invalid tokens line: " + line);
        }

        const std::string token = line.substr(0, split);
        const int id = std::stoi(line.substr(split + 1));
        if (id < 0) {
            throw std::runtime_error("Negative token id in tokens file.");
        }

        if (static_cast<size_t>(id) >= id_to_token.size()) {
            id_to_token.resize(static_cast<size_t>(id) + 1);
        }
        id_to_token[static_cast<size_t>(id)] = token;
    }

    if (id_to_token.empty()) {
        throw std::runtime_error("Tokens file is empty: " + filename.string());
    }

    return id_to_token;
}

int FindBlankId(const std::vector<std::string> &id_to_token) {
    for (size_t i = 0; i < id_to_token.size(); ++i) {
        if (id_to_token[i] == "<blk>" || id_to_token[i] == "<blank>") {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(id_to_token.size()) - 1;
}

bool IsBlankToken(const std::string &token) {
    return token == "<blk>" || token == "<blank>";
}

bool IsSpecialToken(const std::string &token) {
    return IsBlankToken(token) || token == "<unk>";
}

std::string CollapseAsciiWhitespace(const std::string &text) {
    std::string out;
    out.reserve(text.size());

    bool previous_was_space = true;
    for (unsigned char ch : text) {
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
            if (!previous_was_space) {
                out.push_back(' ');
            }
            previous_was_space = true;
            continue;
        }

        out.push_back(static_cast<char>(ch));
        previous_was_space = false;
    }

    if (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

std::string CleanupDecodedTextSpacing(const std::string &text) {
    static const std::string kClosingPunctuation = ",.!?:;)]}";

    std::string out;
    out.reserve(text.size());
    for (unsigned char ch : text) {
        if (!out.empty() &&
            kClosingPunctuation.find(out.back()) != std::string::npos &&
            ch != ' ' &&
            kClosingPunctuation.find(ch) == std::string::npos) {
            out.push_back(' ');
        }
        if (ch == ' ' && !out.empty() && kClosingPunctuation.find(out.back()) != std::string::npos) {
            continue;
        }
        if (kClosingPunctuation.find(ch) != std::string::npos && !out.empty() && out.back() == ' ') {
            out.pop_back();
        }
        out.push_back(static_cast<char>(ch));
    }

    return CollapseAsciiWhitespace(out);
}

std::string PrintableToken(const std::string &token) {
    if (IsSpecialToken(token)) {
        return "";
    }
    if (token == "<space>" || token == "▁") {
        return " ";
    }

    std::string piece = token;
    size_t position = 0;
    while ((position = piece.find("▁", position)) != std::string::npos) {
        piece.replace(position, std::string("▁").size(), " ");
        position += 1;
    }
    return piece;
}

std::string DecodeRnntTokens(const std::vector<int> &tokens, const std::vector<std::string> &id_to_token) {
    std::string text;
    for (int id : tokens) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token.size()) {
            throw std::runtime_error("Decoder produced token id outside vocabulary.");
        }
        text += PrintableToken(id_to_token[static_cast<size_t>(id)]);
    }
    return CleanupDecodedTextSpacing(text);
}

double HzToMelHtk(double freq) {
    return 2595.0 * std::log10(1.0 + freq / 700.0);
}

double MelToHzHtk(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

std::vector<float> BuildMelFilterbank() {
    constexpr int num_freqs = kFftSize / 2 + 1;
    std::vector<float> fbanks(static_cast<size_t>(num_freqs) * kFeatureBins, 0.0f);

    std::vector<double> all_freqs(num_freqs);
    for (int i = 0; i < num_freqs; ++i) {
        all_freqs[static_cast<size_t>(i)] = static_cast<double>(i) * kTargetSampleRate / kFftSize;
    }

    const double mel_min = HzToMelHtk(0.0);
    const double mel_max = HzToMelHtk(8000.0);

    std::vector<double> mel_points(kFeatureBins + 2);
    for (int i = 0; i < kFeatureBins + 2; ++i) {
        const double alpha = static_cast<double>(i) / (kFeatureBins + 1);
        mel_points[static_cast<size_t>(i)] = MelToHzHtk(mel_min + (mel_max - mel_min) * alpha);
    }

    for (int f = 0; f < num_freqs; ++f) {
        const double hz = all_freqs[static_cast<size_t>(f)];
        for (int m = 0; m < kFeatureBins; ++m) {
            const double left = mel_points[static_cast<size_t>(m)];
            const double center = mel_points[static_cast<size_t>(m + 1)];
            const double right = mel_points[static_cast<size_t>(m + 2)];

            double value = 0.0;
            if (hz >= left && hz <= center && center > left) {
                value = (hz - left) / (center - left);
            } else if (hz > center && hz <= right && right > center) {
                value = (right - hz) / (right - center);
            }
            fbanks[static_cast<size_t>(f) * kFeatureBins + m] = static_cast<float>(std::max(0.0, value));
        }
    }

    return fbanks;
}

std::vector<float> BuildPeriodicHannWindow() {
    std::vector<float> window(kFftSize);
    for (int i = 0; i < kFftSize; ++i) {
        window[static_cast<size_t>(i)] = 0.5f - 0.5f * std::cos((2.0f * kPi * i) / kFftSize);
    }
    return window;
}

std::vector<float> ComputeFbank(const std::vector<float> &audio) {
    if (audio.size() < static_cast<size_t>(kFftSize)) {
        throw std::runtime_error("Audio is too short for GigaAM v3 preprocessing.");
    }

    const int64_t num_frames = 1 + static_cast<int64_t>((audio.size() - kFftSize) / kHopSize);
    if (num_frames <= 0) {
        throw std::runtime_error("Feature extractor produced zero frames.");
    }

    const auto window = BuildPeriodicHannWindow();
    const auto mel_fbanks = BuildMelFilterbank();
    constexpr int num_freqs = kFftSize / 2 + 1;

    std::vector<float> cos_table(static_cast<size_t>(num_freqs) * kFftSize);
    std::vector<float> sin_table(static_cast<size_t>(num_freqs) * kFftSize);
    for (int k = 0; k < num_freqs; ++k) {
        for (int n = 0; n < kFftSize; ++n) {
            const float angle = 2.0f * kPi * k * n / kFftSize;
            cos_table[static_cast<size_t>(k) * kFftSize + n] = std::cos(angle);
            sin_table[static_cast<size_t>(k) * kFftSize + n] = std::sin(angle);
        }
    }

    std::vector<float> features(static_cast<size_t>(num_frames) * kFeatureBins);
    std::vector<float> frame(kFftSize);
    std::vector<float> power(num_freqs);

    for (int64_t t = 0; t < num_frames; ++t) {
        const size_t offset = static_cast<size_t>(t) * kHopSize;
        for (int n = 0; n < kFftSize; ++n) {
            frame[static_cast<size_t>(n)] = audio[offset + static_cast<size_t>(n)] * window[static_cast<size_t>(n)];
        }

        for (int k = 0; k < num_freqs; ++k) {
            float re = 0.0f;
            float im = 0.0f;
            for (int n = 0; n < kFftSize; ++n) {
                const float sample = frame[static_cast<size_t>(n)];
                re += sample * cos_table[static_cast<size_t>(k) * kFftSize + n];
                im -= sample * sin_table[static_cast<size_t>(k) * kFftSize + n];
            }
            power[static_cast<size_t>(k)] = re * re + im * im;
        }

        for (int m = 0; m < kFeatureBins; ++m) {
            float mel = 0.0f;
            for (int k = 0; k < num_freqs; ++k) {
                mel += power[static_cast<size_t>(k)] * mel_fbanks[static_cast<size_t>(k) * kFeatureBins + m];
            }
            mel = std::clamp(mel, 1e-9f, 1e9f);
            features[static_cast<size_t>(t) * kFeatureBins + m] = std::log(mel);
        }
    }

    return features;
}

std::vector<float> TransposeFeaturesToBct(const std::vector<float> &features, int64_t num_frames) {
    std::vector<float> transposed(static_cast<size_t>(kFeatureBins * num_frames));
    for (int64_t t = 0; t < num_frames; ++t) {
        for (int64_t c = 0; c < kFeatureBins; ++c) {
            transposed[static_cast<size_t>(c * num_frames + t)] =
                features[static_cast<size_t>(t * kFeatureBins + c)];
        }
    }
    return transposed;
}

EncoderOutput RunEncoder(Ort::Session &encoder,
                         const TensorNameCache &input_names,
                         const TensorNameCache &output_names,
                         const std::vector<float> &features_bct,
                         int64_t num_frames) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> feature_shape = {1, kFeatureBins, num_frames};
    std::vector<int64_t> length_shape = {1};
    int64_t length_value = num_frames;

    Ort::Value features_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float *>(features_bct.data()),
        features_bct.size(),
        feature_shape.data(),
        feature_shape.size());

    Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        &length_value,
        1,
        length_shape.data(),
        length_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(features_tensor));
    inputs.emplace_back(std::move(length_tensor));

    auto outputs = encoder.Run(Ort::RunOptions{nullptr},
                               input_names.ptrs.data(),
                               inputs.data(),
                               inputs.size(),
                               output_names.ptrs.data(),
                               output_names.ptrs.size());

    if (outputs.size() < 2) {
        throw std::runtime_error("RNNT encoder must return encoded tensor and encoded lengths.");
    }

    Ort::Value *encoded_tensor = nullptr;
    Ort::Value *length_tensor_out = nullptr;
    for (auto &output : outputs) {
        auto info = output.GetTensorTypeAndShapeInfo();
        const auto type = info.GetElementType();
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            encoded_tensor = &output;
        } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 || type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            length_tensor_out = &output;
        }
    }

    if (encoded_tensor == nullptr || length_tensor_out == nullptr) {
        throw std::runtime_error("Failed to locate RNNT encoder outputs.");
    }

    int64_t encoded_len = 0;
    {
        auto info = length_tensor_out->GetTensorTypeAndShapeInfo();
        if (info.GetElementCount() < 1) {
            throw std::runtime_error("RNNT encoder length output is empty.");
        }
        if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            encoded_len = length_tensor_out->GetTensorData<int32_t>()[0];
        } else {
            encoded_len = length_tensor_out->GetTensorData<int64_t>()[0];
        }
    }

    auto encoded_info = encoded_tensor->GetTensorTypeAndShapeInfo();
    auto encoded_shape = encoded_info.GetShape();
    if (encoded_shape.size() != 3 || encoded_shape[0] != 1) {
        throw std::runtime_error("Unexpected RNNT encoder output shape.");
    }

    const float *encoded_data = encoded_tensor->GetTensorData<float>();
    EncoderOutput result;
    result.time_steps = encoded_len;

    if (encoded_shape[1] == encoded_len) {
        result.feature_dim = encoded_shape[2];
        result.frames.assign(encoded_data, encoded_data + static_cast<size_t>(encoded_len * result.feature_dim));
    } else if (encoded_shape[2] == encoded_len) {
        result.feature_dim = encoded_shape[1];
        result.frames.resize(static_cast<size_t>(encoded_len * result.feature_dim));
        for (int64_t t = 0; t < encoded_len; ++t) {
            for (int64_t d = 0; d < result.feature_dim; ++d) {
                result.frames[static_cast<size_t>(t * result.feature_dim + d)] =
                    encoded_data[static_cast<size_t>(d * encoded_len + t)];
            }
        }
    } else {
        throw std::runtime_error("Cannot infer RNNT encoder time axis from output shape.");
    }

    return result;
}

size_t DecoderHiddenSize() {
    return 320;
}

Ort::Value CreateTokenTensor(Ort::MemoryInfo &memory_info,
                             ONNXTensorElementDataType element_type,
                             int token,
                             std::vector<int64_t> &shape,
                             int32_t &token_i32,
                             int64_t &token_i64) {
    shape = {1, 1};
    if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        token_i32 = token;
        return Ort::Value::CreateTensor<int32_t>(memory_info, &token_i32, 1, shape.data(), shape.size());
    }

    token_i64 = token;
    return Ort::Value::CreateTensor<int64_t>(memory_info, &token_i64, 1, shape.data(), shape.size());
}

std::vector<float> CopyTensorToVector(const Ort::Value &tensor) {
    auto info = tensor.GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    const float *data = tensor.GetTensorData<float>();
    return std::vector<float>(data, data + count);
}

std::vector<float> PrepareDecoderOutForJoint(const Ort::Value &decoder_out) {
    const float *data = decoder_out.GetTensorData<float>();
    return std::vector<float>(data, data + 320);
}

std::vector<float> RunJoint(Ort::Session &joint,
                            const TensorNameCache &input_names,
                            const TensorNameCache &output_names,
                            const float *encoder_frame,
                            int64_t encoder_dim,
                            const std::vector<float> &decoder_for_joint) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> enc_shape = {1, encoder_dim, 1};
    std::vector<int64_t> dec_shape = {1, static_cast<int64_t>(decoder_for_joint.size()), 1};

    Ort::Value enc_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float *>(encoder_frame),
        static_cast<size_t>(encoder_dim),
        enc_shape.data(),
        enc_shape.size());

    Ort::Value dec_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float *>(decoder_for_joint.data()),
        decoder_for_joint.size(),
        dec_shape.data(),
        dec_shape.size());

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(std::move(enc_tensor));
    inputs.emplace_back(std::move(dec_tensor));

    auto outputs = joint.Run(Ort::RunOptions{nullptr},
                             input_names.ptrs.data(),
                             inputs.data(),
                             inputs.size(),
                             output_names.ptrs.data(),
                             output_names.ptrs.size());

    if (outputs.empty()) {
        throw std::runtime_error("RNNT joint returned no outputs.");
    }
    return CopyTensorToVector(outputs.front());
}

std::vector<int> DecodeRnnt(Ort::Session &decoder,
                            Ort::Session &joint,
                            const TensorNameCache &decoder_input_names,
                            const TensorNameCache &decoder_output_names,
                            const TensorNameCache &joint_input_names,
                            const TensorNameCache &joint_output_names,
                            const EncoderOutput &encoder_out,
                            int blank_id) {
    const size_t hidden_size = DecoderHiddenSize();
    DecoderState state{
        std::vector<float>(hidden_size, 0.0f),
        std::vector<float>(hidden_size, 0.0f),
    };

    auto decoder_input_type = decoder.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int> tokens;
    int64_t t = 0;
    int emitted_tokens = 0;

    while (t < encoder_out.time_steps) {
        std::vector<int64_t> token_shape;
        int32_t token_i32 = 0;
        int64_t token_i64 = 0;
        const int prev_token = tokens.empty() ? blank_id : tokens.back();
        Ort::Value token_tensor = CreateTokenTensor(memory_info, decoder_input_type, prev_token, token_shape, token_i32, token_i64);

        std::vector<int64_t> state_shape = {1, 1, static_cast<int64_t>(hidden_size)};
        Ort::Value h_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            state.h.data(),
            state.h.size(),
            state_shape.data(),
            state_shape.size());
        Ort::Value c_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            state.c.data(),
            state.c.size(),
            state_shape.data(),
            state_shape.size());

        std::vector<Ort::Value> decoder_inputs;
        decoder_inputs.emplace_back(std::move(token_tensor));
        decoder_inputs.emplace_back(std::move(h_tensor));
        decoder_inputs.emplace_back(std::move(c_tensor));

        auto decoder_outputs = decoder.Run(Ort::RunOptions{nullptr},
                                           decoder_input_names.ptrs.data(),
                                           decoder_inputs.data(),
                                           decoder_inputs.size(),
                                           decoder_output_names.ptrs.data(),
                                           decoder_output_names.ptrs.size());

        if (decoder_outputs.size() < 3) {
            throw std::runtime_error("RNNT decoder must return dec, h, c.");
        }

        auto decoder_for_joint = PrepareDecoderOutForJoint(decoder_outputs[0]);
        auto next_h = CopyTensorToVector(decoder_outputs[1]);
        auto next_c = CopyTensorToVector(decoder_outputs[2]);

        const float *encoder_frame = encoder_out.frames.data() + static_cast<size_t>(t * encoder_out.feature_dim);
        auto logits = RunJoint(joint,
                               joint_input_names,
                               joint_output_names,
                               encoder_frame,
                               encoder_out.feature_dim,
                               decoder_for_joint);

        const int token = static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        if (token != blank_id) {
            tokens.push_back(token);
            state.h = std::move(next_h);
            state.c = std::move(next_c);
            ++emitted_tokens;

            if (emitted_tokens >= kMaxTokensPerStep) {
                ++t;
                emitted_tokens = 0;
            }
        } else {
            ++t;
            emitted_tokens = 0;
        }
    }

    return tokens;
}

void PrintUsage(const char *program_name) {
    std::cerr
        << "Usage:\n"
        << "  " << program_name << " infer <audio.wav> [model_dir] [tokens.txt]\n"
        << "  " << program_name << " eval-ru <manifest.tsv> [model_dir] [tokens.txt]\n"
        << "  " << program_name << " <audio.wav> [model_dir] [tokens.txt]\n"
        << "\n"
        << "Default model dir: .\n"
        << "Expected RNNT files in model dir:\n"
        << "  v3_e2e_rnnt_encoder.int8.onnx\n"
        << "  v3_e2e_rnnt_decoder.int8.onnx\n"
        << "  v3_e2e_rnnt_joint.int8.onnx\n"
        << "Default tokens: v3_e2e_rnnt_vocab.txt\n"
        << "\n"
        << "Manifest format for eval-ru:\n"
        << "  path/to/audio.wav<TAB>reference text\n";
}

struct ManifestEntry {
    std::filesystem::path audio_path;
    std::string reference_text;
};

struct EvalTotals {
    size_t utterances = 0;
    size_t word_edits = 0;
    size_t word_total = 0;
    size_t char_edits = 0;
    size_t char_total = 0;
};

class RnntRecognizer {
public:
    RnntRecognizer(const std::filesystem::path &model_dir, const std::filesystem::path &tokens_file)
        : id_to_token_(LoadTokens(tokens_file)),
          blank_id_(FindBlankId(id_to_token_)),
          env_(ORT_LOGGING_LEVEL_WARNING, "gigaam_asr_rnnt") {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetInterOpNumThreads(1);

        const std::filesystem::path encoder_file = model_dir / "v3_e2e_rnnt_encoder.int8.onnx";
        const std::filesystem::path decoder_file = model_dir / "v3_e2e_rnnt_decoder.int8.onnx";
        const std::filesystem::path joint_file = model_dir / "v3_e2e_rnnt_joint.int8.onnx";

        for (const auto &path : {encoder_file, decoder_file, joint_file, tokens_file}) {
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("File not found: " + path.string());
            }
        }

        encoder_ = std::make_unique<Ort::Session>(env_, encoder_file.c_str(), session_options_);
        decoder_ = std::make_unique<Ort::Session>(env_, decoder_file.c_str(), session_options_);
        joint_ = std::make_unique<Ort::Session>(env_, joint_file.c_str(), session_options_);

        encoder_inputs_ = CollectNames(*encoder_, true);
        encoder_outputs_ = CollectNames(*encoder_, false);
        decoder_inputs_ = CollectNames(*decoder_, true);
        decoder_outputs_ = CollectNames(*decoder_, false);
        joint_inputs_ = CollectNames(*joint_, true);
        joint_outputs_ = CollectNames(*joint_, false);
    }

    std::string RecognizeFile(const std::filesystem::path &wav_file) const {
        if (!std::filesystem::exists(wav_file)) {
            throw std::runtime_error("File not found: " + wav_file.string());
        }

        const WavData wav = ReadWavFile(wav_file);
        std::cout << "Loaded audio: " << wav_file << "\n"
                  << "  sample_rate=" << wav.sample_rate << "\n"
                  << "  samples=" << wav.samples.size() << std::endl;

        const std::vector<float> audio_16k = ResampleLinear(wav.samples, wav.sample_rate, kTargetSampleRate);
        std::cout << "After resampling: " << kTargetSampleRate << " Hz, " << audio_16k.size() << " samples" << std::endl;

        const std::vector<float> features = ComputeFbank(audio_16k);
        const int64_t num_frames = static_cast<int64_t>(features.size()) / kFeatureBins;
        const std::vector<float> features_bct = TransposeFeaturesToBct(features, num_frames);
        std::cout << "Features shape: (1, " << kFeatureBins << ", " << num_frames << ")" << std::endl;

        const EncoderOutput encoder_out =
            RunEncoder(*encoder_, encoder_inputs_, encoder_outputs_, features_bct, num_frames);
        std::cout << "Encoder output: time_steps=" << encoder_out.time_steps
                  << ", feature_dim=" << encoder_out.feature_dim << std::endl;

        const std::vector<int> token_ids = DecodeRnnt(*decoder_,
                                                      *joint_,
                                                      decoder_inputs_,
                                                      decoder_outputs_,
                                                      joint_inputs_,
                                                      joint_outputs_,
                                                      encoder_out,
                                                      blank_id_);

        return DecodeRnntTokens(token_ids, id_to_token_);
    }

private:
    std::vector<std::string> id_to_token_;
    int blank_id_ = -1;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> encoder_;
    std::unique_ptr<Ort::Session> decoder_;
    std::unique_ptr<Ort::Session> joint_;
    TensorNameCache encoder_inputs_;
    TensorNameCache encoder_outputs_;
    TensorNameCache decoder_inputs_;
    TensorNameCache decoder_outputs_;
    TensorNameCache joint_inputs_;
    TensorNameCache joint_outputs_;
};

std::string TrimAsciiWhitespace(const std::string &value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> SplitWords(const std::string &text) {
    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

std::vector<uint32_t> Utf8ToCodepoints(const std::string &text) {
    std::vector<uint32_t> result;
    for (size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if ((c & 0x80) == 0) {
            result.push_back(c);
            ++i;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            result.push_back(((c & 0x1F) << 6) |
                             (static_cast<unsigned char>(text[i + 1]) & 0x3F));
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            result.push_back(((c & 0x0F) << 12) |
                             ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                             (static_cast<unsigned char>(text[i + 2]) & 0x3F));
            i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
            result.push_back(((c & 0x07) << 18) |
                             ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
                             ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
                             (static_cast<unsigned char>(text[i + 3]) & 0x3F));
            i += 4;
        } else {
            throw std::runtime_error("Invalid UTF-8 sequence in reference text.");
        }
    }
    return result;
}

std::string CodepointsToUtf8(const std::vector<uint32_t> &codepoints) {
    std::string text;
    for (uint32_t cp : codepoints) {
        if (cp <= 0x7F) {
            text.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            text.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            text.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            text.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            text.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            text.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            text.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            text.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            text.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            text.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return text;
}

uint32_t NormalizeRussianCodepoint(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') {
        return cp + 32;
    }
    if (cp == 0x0401 || cp == 0x0451) {
        return 0x0435;  // ё/Ё -> е
    }
    if (cp >= 0x0410 && cp <= 0x042F) {
        return cp + 32;
    }
    return cp;
}

bool IsNormalizedWordCodepoint(uint32_t cp) {
    return (cp >= 'a' && cp <= 'z') ||
           (cp >= '0' && cp <= '9') ||
           (cp >= 0x0430 && cp <= 0x044F);
}

std::string NormalizeTextForMetrics(const std::string &text) {
    const auto codepoints = Utf8ToCodepoints(text);

    std::vector<uint32_t> normalized;
    normalized.reserve(codepoints.size());

    bool pending_space = false;
    for (uint32_t cp : codepoints) {
        cp = NormalizeRussianCodepoint(cp);
        if (IsNormalizedWordCodepoint(cp)) {
            if (pending_space && !normalized.empty()) {
                normalized.push_back(' ');
            }
            normalized.push_back(cp);
            pending_space = false;
        } else {
            pending_space = true;
        }
    }

    return CodepointsToUtf8(normalized);
}

template <typename T>
size_t EditDistance(const std::vector<T> &a, const std::vector<T> &b) {
    std::vector<size_t> prev(b.size() + 1);
    std::vector<size_t> curr(b.size() + 1);

    for (size_t j = 0; j <= b.size(); ++j) {
        prev[j] = j;
    }

    for (size_t i = 1; i <= a.size(); ++i) {
        curr[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            const size_t cost = a[i - 1] == b[j - 1] ? 0 : 1;
            curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
        }
        std::swap(prev, curr);
    }

    return prev[b.size()];
}

std::vector<ManifestEntry> LoadManifest(const std::filesystem::path &manifest_file) {
    std::ifstream input(manifest_file);
    if (!input) {
        throw std::runtime_error("Cannot open manifest file: " + manifest_file.string());
    }

    const std::filesystem::path base_dir = manifest_file.parent_path();
    std::vector<ManifestEntry> entries;
    std::string line;
    size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        const size_t tab_pos = line.find('\t');
        if (tab_pos == std::string::npos) {
            throw std::runtime_error("Manifest line must contain a TAB separator at line " + std::to_string(line_number));
        }

        std::filesystem::path audio_path = line.substr(0, tab_pos);
        if (audio_path.is_relative()) {
            audio_path = base_dir / audio_path;
        }

        entries.push_back({audio_path, TrimAsciiWhitespace(line.substr(tab_pos + 1))});
    }

    if (entries.empty()) {
        throw std::runtime_error("Manifest is empty: " + manifest_file.string());
    }

    return entries;
}

void UpdateTotals(EvalTotals &totals, const std::string &reference, const std::string &hypothesis) {
    const std::string normalized_reference = NormalizeTextForMetrics(reference);
    const std::string normalized_hypothesis = NormalizeTextForMetrics(hypothesis);

    const auto ref_words = SplitWords(normalized_reference);
    const auto hyp_words = SplitWords(normalized_hypothesis);
    const auto ref_chars = Utf8ToCodepoints(normalized_reference);
    const auto hyp_chars = Utf8ToCodepoints(normalized_hypothesis);

    totals.utterances += 1;
    totals.word_edits += EditDistance(ref_words, hyp_words);
    totals.word_total += ref_words.size();
    totals.char_edits += EditDistance(ref_chars, hyp_chars);
    totals.char_total += ref_chars.size();
}

int RunInferMode(const std::filesystem::path &wav_file,
                 const std::filesystem::path &model_dir,
                 const std::filesystem::path &tokens_file) {
    RnntRecognizer recognizer(model_dir, tokens_file);
    const std::string text = recognizer.RecognizeFile(wav_file);
    std::cout << "Recognized text: " << text << std::endl;
    return 0;
}

int RunEvalMode(const std::filesystem::path &manifest_file,
                const std::filesystem::path &model_dir,
                const std::filesystem::path &tokens_file) {
    const auto entries = LoadManifest(manifest_file);
    RnntRecognizer recognizer(model_dir, tokens_file);
    EvalTotals totals;

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto &entry = entries[i];
        std::cout << "[" << (i + 1) << "/" << entries.size() << "] " << entry.audio_path << std::endl;
        const std::string hypothesis = recognizer.RecognizeFile(entry.audio_path);
        UpdateTotals(totals, entry.reference_text, hypothesis);
        std::cout << "Reference:  " << entry.reference_text << std::endl;
        std::cout << "Recognized: " << hypothesis << std::endl;
    }

    const double wer = totals.word_total == 0 ? 0.0 : 100.0 * static_cast<double>(totals.word_edits) / totals.word_total;
    const double cer = totals.char_total == 0 ? 0.0 : 100.0 * static_cast<double>(totals.char_edits) / totals.char_total;

    std::cout << "Summary:" << std::endl;
    std::cout << "  utterances=" << totals.utterances << std::endl;
    std::cout << "  WER=" << wer << "% (" << totals.word_edits << "/" << totals.word_total << ")" << std::endl;
    std::cout << "  CER=" << cer << "% (" << totals.char_edits << "/" << totals.char_total << ")" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char *argv[]) {
    try {
        if (argc < 2) {
            PrintUsage(argv[0]);
            return 1;
        }

        const std::string mode = argv[1];
        if (mode == "infer") {
            if (argc < 3 || argc > 5) {
                PrintUsage(argv[0]);
                return 1;
            }
            const std::filesystem::path wav_file = argv[2];
            const std::filesystem::path model_dir = argc >= 4 ? std::filesystem::path(argv[3]) : std::filesystem::path(".");
            const std::filesystem::path tokens_file =
                argc >= 5 ? std::filesystem::path(argv[4]) : std::filesystem::path("v3_e2e_rnnt_vocab.txt");
            return RunInferMode(wav_file, model_dir, tokens_file);
        }

        if (mode == "eval-ru") {
            if (argc < 3 || argc > 5) {
                PrintUsage(argv[0]);
                return 1;
            }
            const std::filesystem::path manifest_file = argv[2];
            const std::filesystem::path model_dir = argc >= 4 ? std::filesystem::path(argv[3]) : std::filesystem::path(".");
            const std::filesystem::path tokens_file =
                argc >= 5 ? std::filesystem::path(argv[4]) : std::filesystem::path("v3_e2e_rnnt_vocab.txt");
            return RunEvalMode(manifest_file, model_dir, tokens_file);
        }

        if (argc > 4) {
            PrintUsage(argv[0]);
            return 1;
        }

        const std::filesystem::path wav_file = argv[1];
        const std::filesystem::path model_dir = argc >= 3 ? std::filesystem::path(argv[2]) : std::filesystem::path(".");
        const std::filesystem::path tokens_file =
            argc >= 4 ? std::filesystem::path(argv[3]) : std::filesystem::path("v3_e2e_rnnt_vocab.txt");
        return RunInferMode(wav_file, model_dir, tokens_file);
    } catch (const Ort::Exception &e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 2;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
