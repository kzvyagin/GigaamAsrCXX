#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "domain/Assets.hpp"
#include "domain/Manifest.hpp"
#include "domain/Text.hpp"

namespace {

void Expect(bool condition, const std::string &message) {
    if (!condition) {
        std::cerr << "Test failure: " << message << '\n';
        std::exit(1);
    }
}

void TestTextNormalization() {
    const std::string normalized = gigaam::domain::NormalizeTextForMetrics("Примите, Ёж!");
    Expect(normalized == "примите еж", "NormalizeTextForMetrics must lowercase and normalize punctuation.");
}

void TestEvalTotals() {
    gigaam::domain::EvalTotals totals;
    gigaam::domain::UpdateTotals(totals, "привет мир", "Привет, мир!");
    Expect(totals.word_edits == 0, "Word edits should be zero for normalized equivalent text.");
    Expect(totals.char_edits == 0, "Char edits should be zero for normalized equivalent text.");
}

void TestModelAssets() {
    const auto assets = gigaam::domain::DefaultE2eModelAssets();
    Expect(assets.size() == 4, "E2E model catalog must expose 4 assets.");
    Expect(assets.front().file_name == "v3_e2e_rnnt_encoder.int8.onnx", "Unexpected first model asset.");
}

void TestManifestRoundTrip() {
    const auto temp_dir = std::filesystem::temp_directory_path() / "gigaam_domain_tests";
    std::filesystem::create_directories(temp_dir);
    const auto manifest = temp_dir / "manifest.tsv";

    gigaam::domain::WriteManifest(manifest, {
        {std::filesystem::path("audio") / "sample_001.wav", "привет мир"},
        {std::filesystem::path("audio") / "sample_002.wav", "ещё тест"},
    });

    const auto entries = gigaam::domain::LoadManifest(manifest);
    Expect(entries.size() == 2, "Manifest round-trip must preserve number of entries.");
    Expect(entries[0].reference_text == "привет мир", "Manifest round-trip must preserve text.");
}

}  // namespace

int main() {
    TestTextNormalization();
    TestEvalTotals();
    TestModelAssets();
    TestManifestRoundTrip();
    return 0;
}
