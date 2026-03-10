#include <filesystem>
#include <iostream>

#include <CLI/CLI.hpp>

#include "app/Services.hpp"
#include "domain/Assets.hpp"
#include "domain/Dataset.hpp"
#include "domain/Text.hpp"
#include "infra/asr/RnntRecognizer.hpp"
#include "infra/audio/MiniaudioDecoder.hpp"
#include "infra/datasets/RussianLibrispeechIndex.hpp"
#include "infra/download/HttplibHttpClient.hpp"

namespace {

void PrintEvalReport(const gigaam::app::EvalReport &report, bool verbose) {
    if (verbose) {
        for (size_t i = 0; i < report.items.size(); ++i) {
            const auto &item = report.items[i];
            std::cout << "[" << (i + 1) << "/" << report.items.size() << "] " << item.audio_path << '\n';
            std::cout << "Reference:  " << item.reference_text << '\n';
            std::cout << "Recognized: " << item.hypothesis_text << '\n';
        }
    }

    std::cout << "Summary:\n";
    std::cout << "  utterances=" << report.totals.utterances << '\n';
    std::cout << "  WER=" << gigaam::domain::ComputeWerPercent(report.totals) << "% ("
              << report.totals.word_edits << "/" << report.totals.word_total << ")\n";
    std::cout << "  CER=" << gigaam::domain::ComputeCerPercent(report.totals) << "% ("
              << report.totals.char_edits << "/" << report.totals.char_total << ")\n";
}

}  // namespace

int main(int argc, char *argv[]) {
    try {
        CLI::App app("GigaAM v3 E2E RNN-T CLI");
        app.require_subcommand(1);

        gigaam::infra::download::HttplibHttpClient http_client;
        gigaam::infra::audio::MiniaudioDecoder audio_decoder;

        auto *models = app.add_subcommand("models", "Model asset operations");
        auto *models_download = models->add_subcommand("download", "Download E2E model artifacts");
        std::filesystem::path models_output_dir = gigaam::domain::DefaultModelDirectory();
        bool models_force = false;
        models_download->add_option("--output-dir", models_output_dir, "Output directory for model files");
        models_download->add_flag("--force", models_force, "Overwrite existing files");
        models_download->callback([&]() {
            gigaam::app::ModelDownloadService service(http_client);
            service.Run(models_output_dir, models_force, std::cout);
        });

        auto *datasets = app.add_subcommand("datasets", "Dataset operations");
        auto *datasets_download = datasets->add_subcommand("download", "Download public Russian evaluation samples");
        std::string dataset_name = "ru-public";
        std::filesystem::path dataset_output_dir = gigaam::domain::DefaultDatasetDirectory();
        size_t dataset_limit = 10;
        bool datasets_force = false;
        datasets_download->add_option("dataset", dataset_name, "Dataset name")->default_val("ru-public");
        datasets_download->add_option("--output-dir", dataset_output_dir, "Output directory for dataset files");
        datasets_download->add_option("--limit", dataset_limit, "How many public samples to download")->check(CLI::PositiveNumber);
        datasets_download->add_flag("--force", datasets_force, "Overwrite existing files");
        datasets_download->callback([&]() {
            if (dataset_name != "ru-public") {
                throw std::runtime_error("Only dataset 'ru-public' is supported.");
            }

            gigaam::infra::datasets::RussianLibrispeechIndex index(http_client);
            gigaam::app::DatasetDownloadService service(http_client, index);
            service.Run(dataset_output_dir, dataset_limit, datasets_force, std::cout);
        });

        auto *infer = app.add_subcommand("infer", "Run inference for a single audio file");
        std::filesystem::path infer_audio_file;
        std::filesystem::path infer_model_dir = gigaam::domain::DefaultModelDirectory();
        infer->add_option("audio-file", infer_audio_file, "Audio file to transcribe")->required();
        infer->add_option("--model-dir", infer_model_dir, "Directory with E2E model files");
        infer->callback([&]() {
            gigaam::infra::asr::RnntRecognizer recognizer(gigaam::domain::ResolveModelLayout(infer_model_dir));
            gigaam::app::InferenceService service(audio_decoder, recognizer);
            std::cout << service.Run(infer_audio_file) << '\n';
        });

        auto *eval = app.add_subcommand("eval", "Run evaluation for a TSV manifest");
        std::filesystem::path eval_manifest_file;
        std::filesystem::path eval_model_dir = gigaam::domain::DefaultModelDirectory();
        bool eval_verbose = false;
        eval->add_option("manifest", eval_manifest_file, "Manifest TSV path")->required();
        eval->add_option("--model-dir", eval_model_dir, "Directory with E2E model files");
        eval->add_flag("--verbose", eval_verbose, "Print per-file reference and hypothesis");
        eval->callback([&]() {
            gigaam::infra::asr::RnntRecognizer recognizer(gigaam::domain::ResolveModelLayout(eval_model_dir));
            gigaam::app::InferenceService inference(audio_decoder, recognizer);
            gigaam::app::EvaluationService evaluation(inference);
            PrintEvalReport(evaluation.Run(eval_manifest_file), eval_verbose);
        });

        CLI11_PARSE(app, argc, argv);
        return 0;
    } catch (const Ort::Exception &e) {
        std::cerr << "ONNX Runtime error: " << e.what() << '\n';
        return 2;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
