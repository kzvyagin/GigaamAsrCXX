#include "infra/asr/RnntRecognizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace gigaam::infra::asr {

namespace {

constexpr int kTargetSampleRate = 16000;
constexpr int kFeatureBins = 64;
constexpr int kFftSize = 320;
constexpr int kHopSize = 160;
constexpr int kMaxTokensPerStep = 3;
constexpr float kPi = 3.14159265358979323846f;

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

RnntRecognizer::TensorNameCache CollectNames(Ort::Session &session, bool inputs) {
    RnntRecognizer::TensorNameCache cache;
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
            sin_table[static_cast<size_t>(k) * kFftSize + n] = -std::sin(angle);
        }
    }

    std::vector<float> features(static_cast<size_t>(num_frames) * kFeatureBins);
    std::vector<float> frame(kFftSize);
    std::vector<float> power(num_freqs);

    for (int64_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        const size_t offset = static_cast<size_t>(frame_idx) * kHopSize;
        for (int i = 0; i < kFftSize; ++i) {
            frame[static_cast<size_t>(i)] = audio[offset + static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
        }

        for (int k = 0; k < num_freqs; ++k) {
            float real = 0.0f;
            float imag = 0.0f;
            for (int n = 0; n < kFftSize; ++n) {
                const float sample = frame[static_cast<size_t>(n)];
                real += sample * cos_table[static_cast<size_t>(k) * kFftSize + n];
                imag += sample * sin_table[static_cast<size_t>(k) * kFftSize + n];
            }
            power[static_cast<size_t>(k)] = real * real + imag * imag;
        }

        for (int bin = 0; bin < kFeatureBins; ++bin) {
            double energy = 1.0e-10;
            for (int k = 0; k < num_freqs; ++k) {
                energy += power[static_cast<size_t>(k)] *
                          mel_fbanks[static_cast<size_t>(k) * kFeatureBins + bin];
            }
            features[static_cast<size_t>(frame_idx) * kFeatureBins + bin] = static_cast<float>(std::log(energy));
        }
    }

    return features;
}

std::vector<float> TransposeFeaturesToBct(const std::vector<float> &features, int64_t num_frames) {
    std::vector<float> transposed(static_cast<size_t>(num_frames) * kFeatureBins);
    for (int64_t t = 0; t < num_frames; ++t) {
        for (int c = 0; c < kFeatureBins; ++c) {
            transposed[static_cast<size_t>(c) * num_frames + static_cast<size_t>(t)] =
                features[static_cast<size_t>(t) * kFeatureBins + c];
        }
    }
    return transposed;
}

std::vector<float> CopyTensorToVector(const Ort::Value &value) {
    const auto info = value.GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    const float *data = value.GetTensorData<float>();
    return std::vector<float>(data, data + count);
}

RnntRecognizer::EncoderOutput RunEncoder(Ort::Session &encoder,
                                         const RnntRecognizer::TensorNameCache &input_names,
                                         const RnntRecognizer::TensorNameCache &output_names,
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
    RnntRecognizer::EncoderOutput result;
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

std::vector<float> PrepareDecoderOutForJoint(const Ort::Value &decoder_out) {
    const float *data = decoder_out.GetTensorData<float>();
    return std::vector<float>(data, data + 320);
}

std::vector<float> RunJoint(Ort::Session &joint,
                            const RnntRecognizer::TensorNameCache &input_names,
                            const RnntRecognizer::TensorNameCache &output_names,
                            const float *encoder_frame,
                            int64_t encoder_dim,
                            const std::vector<float> &decoder_for_joint) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> enc_shape = {1, encoder_dim, 1};
    std::vector<int64_t> dec_shape = {1, static_cast<int64_t>(decoder_for_joint.size()), 1};

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(encoder_frame), static_cast<size_t>(encoder_dim), enc_shape.data(), enc_shape.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(decoder_for_joint.data()), decoder_for_joint.size(), dec_shape.data(), dec_shape.size()));

    auto outputs = joint.Run(Ort::RunOptions{nullptr},
                             input_names.ptrs.data(),
                             inputs.data(),
                             inputs.size(),
                             output_names.ptrs.data(),
                             output_names.ptrs.size());

    if (outputs.empty()) {
        throw std::runtime_error("RNNT joint returned no outputs.");
    }

    return CopyTensorToVector(outputs[0]);
}

std::vector<int> DecodeRnnt(Ort::Session &decoder,
                            Ort::Session &joint,
                            const RnntRecognizer::TensorNameCache &decoder_input_names,
                            const RnntRecognizer::TensorNameCache &decoder_output_names,
                            const RnntRecognizer::TensorNameCache &joint_input_names,
                            const RnntRecognizer::TensorNameCache &joint_output_names,
                            const RnntRecognizer::EncoderOutput &encoder_out,
                            int blank_id) {
    const size_t hidden_size = DecoderHiddenSize();
    RnntRecognizer::DecoderState state{
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
            memory_info, state.h.data(), state.h.size(), state_shape.data(), state_shape.size());
        Ort::Value c_tensor = Ort::Value::CreateTensor<float>(
            memory_info, state.c.data(), state.c.size(), state_shape.data(), state_shape.size());

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

}  // namespace

RnntRecognizer::RnntRecognizer(const domain::ModelLayout &model_layout)
    : id_to_token_(LoadTokens(model_layout.vocab)),
      blank_id_(FindBlankId(id_to_token_)),
      env_(ORT_LOGGING_LEVEL_WARNING, "gigaam_e2e_rnnt") {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetInterOpNumThreads(1);

    for (const auto &path : {model_layout.encoder, model_layout.decoder, model_layout.joint, model_layout.vocab}) {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("File not found: " + path.string());
        }
    }

    encoder_ = std::make_unique<Ort::Session>(env_, model_layout.encoder.c_str(), session_options_);
    decoder_ = std::make_unique<Ort::Session>(env_, model_layout.decoder.c_str(), session_options_);
    joint_ = std::make_unique<Ort::Session>(env_, model_layout.joint.c_str(), session_options_);

    encoder_inputs_ = CollectNames(*encoder_, true);
    encoder_outputs_ = CollectNames(*encoder_, false);
    decoder_inputs_ = CollectNames(*decoder_, true);
    decoder_outputs_ = CollectNames(*decoder_, false);
    joint_inputs_ = CollectNames(*joint_, true);
    joint_outputs_ = CollectNames(*joint_, false);
}

std::string RnntRecognizer::Recognize(const domain::AudioBuffer &audio) const {
    const auto audio_16k = ResampleLinear(audio.samples, audio.sample_rate, kTargetSampleRate);
    const auto features = ComputeFbank(audio_16k);
    const int64_t num_frames = static_cast<int64_t>(features.size()) / kFeatureBins;
    const auto features_bct = TransposeFeaturesToBct(features, num_frames);

    const auto encoder_out = RunEncoder(*encoder_, encoder_inputs_, encoder_outputs_, features_bct, num_frames);
    const auto token_ids = DecodeRnnt(*decoder_,
                                      *joint_,
                                      decoder_inputs_,
                                      decoder_outputs_,
                                      joint_inputs_,
                                      joint_outputs_,
                                      encoder_out,
                                      blank_id_);
    return DecodeRnntTokens(token_ids, id_to_token_);
}

}  // namespace gigaam::infra::asr
