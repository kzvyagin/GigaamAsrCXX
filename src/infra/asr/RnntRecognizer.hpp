#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "app/Ports.hpp"
#include "domain/Assets.hpp"

namespace gigaam::infra::asr {

class RnntRecognizer : public app::IRecognizer {
public:
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

    explicit RnntRecognizer(const domain::ModelLayout &model_layout);
    std::string Recognize(const domain::AudioBuffer &audio) const override;

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

}  // namespace gigaam::infra::asr
