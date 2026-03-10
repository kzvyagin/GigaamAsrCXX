#include "domain/Assets.hpp"

namespace gigaam::domain {

namespace {

constexpr char kModelBaseUrl[] = "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/";

std::string MakeModelUrl(const std::string &file_name) {
    return std::string(kModelBaseUrl) + file_name;
}

}  // namespace

std::vector<ModelAsset> DefaultE2eModelAssets() {
    return {
        {"v3_e2e_rnnt_encoder.int8.onnx", MakeModelUrl("v3_e2e_rnnt_encoder.int8.onnx")},
        {"v3_e2e_rnnt_decoder.int8.onnx", MakeModelUrl("v3_e2e_rnnt_decoder.int8.onnx")},
        {"v3_e2e_rnnt_joint.int8.onnx", MakeModelUrl("v3_e2e_rnnt_joint.int8.onnx")},
        {"v3_e2e_rnnt_vocab.txt", MakeModelUrl("v3_e2e_rnnt_vocab.txt")},
    };
}

ModelLayout ResolveModelLayout(const std::filesystem::path &model_dir) {
    return {
        model_dir,
        model_dir / "v3_e2e_rnnt_encoder.int8.onnx",
        model_dir / "v3_e2e_rnnt_decoder.int8.onnx",
        model_dir / "v3_e2e_rnnt_joint.int8.onnx",
        model_dir / "v3_e2e_rnnt_vocab.txt",
    };
}

std::filesystem::path DefaultModelDirectory() {
    return std::filesystem::path("models") / "gigaam-v3-e2e-rnnt";
}

}  // namespace gigaam::domain
