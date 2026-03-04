#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <onnxruntime_cxx_api.h>

// Для чтения WAV (одна заголовочный файл)
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

// Для извлечения признаков (Kaldi Native Fbank)
#include "kaldi-native-fbank/csrc/online-feature.h"

// Для ресемплинга (опционально)
#include <samplerate.h>

// -----------------------------------------------------------------------------
// Загрузка словаря tokens.txt
std::vector<std::string> load_tokens(const std::string& filename) {
    std::vector<std::string> id2token;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        // Удаляем возможный символ возврата каретки
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t space_pos = line.find(' ');
        if (space_pos == std::string::npos) {
            // Только число (пробел)
            int id = std::stoi(line);
            if (id >= static_cast<int>(id2token.size())) {
                id2token.resize(id + 1);
            }
            id2token[id] = " ";
        } else {
            std::string token = line.substr(0, space_pos);
            int id = std::stoi(line.substr(space_pos + 1));
            if (id >= static_cast<int>(id2token.size())) {
                id2token.resize(id + 1);
            }
            id2token[id] = token;
        }
    }
    return id2token;
}

 drwav_allocation_callbacks allocation_callbacks_or_defaults(const drwav_allocation_callbacks* pAllocationCallbacks)
{
    if (pAllocationCallbacks != NULL) {
        /* Copy. */
        return *pAllocationCallbacks;
    } else {
        /* Defaults. */
        drwav_allocation_callbacks allocationCallbacks;
        allocationCallbacks.pUserData = NULL;
        allocationCallbacks.onMalloc  = drwav__malloc_default;
        allocationCallbacks.onRealloc = drwav__realloc_default;
        allocationCallbacks.onFree    = drwav__free_default;
        return allocationCallbacks;
    }
}
// -----------------------------------------------------------------------------
// Чтение WAV-файла в моно, 16 кГц, float32
bool read_wav(const std::string& filename, std::vector<float>& audio, int& sample_rate) {
    unsigned int channels;
    unsigned int sample_rate_uint;
    drwav_uint64 total_pcm_frame_count;

    float* samples = drwav_open_file_and_read_pcm_frames_f32(
        filename.c_str(), &channels, &sample_rate_uint, &total_pcm_frame_count, nullptr);
    if (!samples) {
        std::cerr << "Failed to read WAV file." << std::endl;
        return false;
    }

    sample_rate = static_cast<int>(sample_rate_uint);
    // Преобразуем в моно, если нужно (берём среднее по каналам)
    if (channels == 1) {
        audio.assign(samples, samples + total_pcm_frame_count);
    } else {
        audio.resize(total_pcm_frame_count);
        for (drwav_uint64 i = 0; i < total_pcm_frame_count; ++i) {
            float sum = 0.0f;
            for (unsigned int c = 0; c < channels; ++c) {
                sum += samples[i * channels + c];
            }
            audio[i] = sum / channels;
        }
    }
    drwav_free( samples, nullptr );
    return true;
}

// -----------------------------------------------------------------------------
// Ресемплинг до 16 кГц, если необходимо
bool resample_to_16k(const std::vector<float>& input, int input_sr,
                     std::vector<float>& output, int output_sr = 16000) {
    if (input_sr == output_sr) {
        output = input;
        return true;
    }
    double src_ratio = static_cast<double>(output_sr) / input_sr;
    long input_frames = static_cast<long>(input.size());
    long output_frames = static_cast<long>(std::ceil(input_frames * src_ratio));

    output.resize(output_frames);

    SRC_DATA src_data;
    src_data.data_in = input.data();
    src_data.data_out = output.data();
    src_data.input_frames = input_frames;
    src_data.output_frames = output_frames;
    src_data.src_ratio = src_ratio;

    int error = src_simple(&src_data, SRC_SINC_BEST_QUALITY, 1);
    if (error != 0) {
        std::cerr << "Resampling error: " << src_strerror(error) << std::endl;
        return false;
    }
    output.resize(src_data.output_frames_gen);
    return true;
}

// -----------------------------------------------------------------------------
// Извлечение признаков с помощью Kaldi Native Fbank
std::vector<float> compute_fbank(const std::vector<float>& audio, int sample_rate) {
    knf::FbankOptions opts;
    opts.frame_opts.dither = 0.0;
    opts.frame_opts.remove_dc_offset = false;
    opts.frame_opts.preemph_coeff = 0.0;
    opts.frame_opts.window_type = "hann";
    opts.frame_opts.round_to_power_of_two = false;
    opts.mel_opts.low_freq = 0;
    opts.mel_opts.high_freq = 8000;
    opts.mel_opts.num_bins = 64;

    knf::OnlineFbank fbank(opts);
    fbank.AcceptWaveform(sample_rate, audio.data(), audio.size());
    fbank.InputFinished();

    int32_t num_frames = fbank.NumFramesReady();
    int32_t feat_dim = opts.mel_opts.num_bins;

    std::vector<float> features(num_frames * feat_dim);
    for (int32_t i = 0; i < num_frames; ++i) {
        const float* frame = fbank.GetFrame(i);
        std::memcpy(features.data() + i * feat_dim, frame, feat_dim * sizeof(float));
    }
    return features;
}

// -----------------------------------------------------------------------------
// Жадный CTC декодер
std::string ctc_greedy_decode(const float* logits, int seq_len, int vocab_size,
                              const std::vector<std::string>& id2token, int blank_id) {
    std::string result;
    int prev_token = -1;
    for (int t = 0; t < seq_len; ++t) {
        int max_id = 0;
        float max_val = logits[t * vocab_size];
        for (int c = 1; c < vocab_size; ++c) {
            if (logits[t * vocab_size + c] > max_val) {
                max_val = logits[t * vocab_size + c];
                max_id = c;
            }
        }
        if (max_id != blank_id && max_id != prev_token) {
            result += id2token[max_id];
        }
        prev_token = (max_id == blank_id) ? -1 : max_id;
    }
    return result;
}

// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <audio.wav>" << std::endl;
        return 1;
    }
    std::string wav_file = argv[1];
    const std::string model_file = "v3_e2e_ctc.int8.onnx";// "model.int8.onnx";
    const std::string tokens_file = "v3_vocab.txt"; //"tokens.txt";

    // 1. Загрузка словаря
    auto id2token = load_tokens(tokens_file);
    int blank_id = id2token.size() - 1; // последний токен — blank

    // 2. Инициализация ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test-onnx-ctc");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    Ort::Session session(env, model_file.c_str(), session_options);

    // Вывод информации о входных/выходных тензорах (для отладки)
    std::cout << "========== Inputs ==========" << std::endl;
    for (size_t i = 0; i < session.GetInputCount(); ++i) {
        auto name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        auto type_info = session.GetInputTypeInfo(i);
        auto shape_info = type_info.GetTensorTypeAndShapeInfo();
        std::cout << name.get() << " : shape = ";
        for (auto dim : shape_info.GetShape()) std::cout << dim << " ";
        std::cout << std::endl;
    }
    std::cout << "========== Outputs ==========" << std::endl;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        auto name = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        auto type_info = session.GetOutputTypeInfo(i);
        auto shape_info = type_info.GetTensorTypeAndShapeInfo();
        std::cout << name.get() << " : shape = ";
        for (auto dim : shape_info.GetShape()) std::cout << dim << " ";
        std::cout << std::endl;
    }

    // 3. Чтение и предобработка аудио
    std::vector<float> audio;
    int sample_rate;
    if (!read_wav(wav_file, audio, sample_rate)) {
        return 1;
    }
    std::cout << "Original sample rate: " << sample_rate << ", channels: 1 (after mixdown), length: " << audio.size() << " samples" << std::endl;

    // Ресемплинг до 16 кГц
    std::vector<float> audio_16k;
    if (!resample_to_16k(audio, sample_rate, audio_16k, 16000)) {
        return 1;
    }
    std::cout << "After resampling: 16000 Hz, " << audio_16k.size() << " samples" << std::endl;

    // 4. Извлечение признаков
    auto features = compute_fbank(audio_16k, 16000);
    int num_frames = features.size() / 64; // 64 — число фильтров
    std::cout << "Features shape: (" << num_frames << ", 64)" << std::endl;

    // 5. Подготовка тензоров для модели
    // Модель ожидает: вход 'audio_signal' формы [batch, 64, time] и 'length' формы [batch]
    // В Python делали: x (T, C) -> транспонирование в (C, T) -> unsqueeze(0) -> (1, C, T)
    // Мы сразу подготовим данные в нужном порядке.
    std::vector<int64_t> input_shape = {1, 64, num_frames};
    std::vector<float> input_tensor_values(num_frames * 64);
    // Транспонируем из (T, C) в (C, T) (так как Kaldi выдаёт (frames, features))
    for (int t = 0; t < num_frames; ++t) {
        for (int c = 0; c < 64; ++c) {
            input_tensor_values[c * num_frames + t] = features[t * 64 + c];
        }
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    int64_t length_data = num_frames;
    std::vector<int64_t> length_shape = {1};
    Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, &length_data, 1, length_shape.data(), length_shape.size());

    // Имена входов (должны совпадать с тем, что вывелось выше)
    const char* input_names[] = {"audio_signal", "length"};
    const char* output_names[] = {"logprobs"}; // предположительно
/*
    const std::string model_path = "gigaam-v3-ctc.onnx"; // путь к вашей модели
    Ort::Session session(env, model_path.c_str(), session_options);
            auto outputTensors = m_session->Run(
            Ort::RunOptions{nullptr},
            inputNamesToUse.data(),
            ortInputs.data(),
            ortInputs.size(),
            m_outputNamePtrs.data(),
            m_outputNamePtrs.size()
        );
*/
    // 6. Инференс
    std::vector<Ort::Value>  output_tensors;/* = session.Run(Ort::RunOptions{nullptr},
                                       input_names,  &input_tensor,  1, // первый вход — audio_signal
                                       input_names+1, &length_tensor, 1, // второй вход — length
                                       output_names, 1);*/

    // 7. Обработка выхода
    Ort::Value& output = output_tensors.front();
    auto output_info = output.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape(); // ожидается [1, time_out, vocab_size]
    if (output_shape.size() != 3) {
        std::cerr << "Unexpected output shape" << std::endl;
        return 1;
    }
    int64_t out_seq_len = output_shape[1];
    int64_t vocab_size = output_shape[2];
    const float* output_data = output.GetTensorData<float>();

    std::string text = ctc_greedy_decode(output_data, out_seq_len, vocab_size, id2token, blank_id);
    std::cout << "Recognized text: " << text << std::endl;

    return 0;
}