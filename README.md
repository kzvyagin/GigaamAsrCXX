Пример STT для Linux на C++ с моделью GigaAM v3 RNN-T через ONNX Runtime.

Текущая версия примера ориентирована на:

- `v3_rnnt_encoder.int8.onnx`
- `v3_rnnt_decoder.int8.onnx`
- `v3_rnnt_joint.int8.onnx`
- `v3_vocab.txt`

Это практичнее, чем старый минимальный CTC-вариант: в статье на Habr лучшие результаты были именно у `gigaam-v3-e2e-rnnt`, а обычный `v3_rnnt` остаётся самым удобным путём для небольшого C++-демо, потому что у него есть явный символьный словарь без внешнего токенизатора.

Источники:

- GigaAM: https://github.com/salute-developers/GigaAM
- ONNX-конвертация GigaAM v3: https://huggingface.co/istupakov/gigaam-v3-onnx
- Сравнение моделей: https://habr.com/ru/articles/1002260/

## Что внутри

- загрузка WAV;
- ресемплинг в `16 kHz`;
- вычисление `64-bin log-mel` признаков прямо в C++;
- greedy RNN-T decoding через три ONNX-модели: `encoder`, `decoder`, `joint`.

`kaldi-native-fbank` больше не нужен: препроцессор `gigaam-v3` реализован прямо в `main.cpp`.

## Сборка

По умолчанию ONNX Runtime скачивается через `FetchContent` как готовый бинарный архив:

```bash
cmake -S . -B build
cmake --build build -j
```

По умолчанию используется `ONNXRUNTIME_VERSION=1.18.1`. Если нужна другая версия или свой архив:

```bash
cmake -S . -B build -DONNXRUNTIME_VERSION=1.18.1
```

Или явно свой URL:

```bash
cmake -S . -B build \
  -DONNXRUNTIME_URL=https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz
```

Если автозагрузка не нужна:

```bash
cmake -S . -B build \
  -DGIGAAM_FETCH_ONNXRUNTIME=OFF \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.xx.x
```

Ожидаемая структура локального ONNX Runtime:

- `ONNXRUNTIME_ROOT/include`
- `ONNXRUNTIME_ROOT/lib/libonnxruntime.so`

## Модели

Скачать ONNX-артефакты можно из:

https://huggingface.co/istupakov/gigaam-v3-onnx

Нужны файлы:

- `v3_rnnt_encoder.int8.onnx`
- `v3_rnnt_decoder.int8.onnx`
- `v3_rnnt_joint.int8.onnx`
- `v3_vocab.txt`

## Режимы

### 1. Прогон одного WAV

Если файлы лежат рядом с бинарником или в текущей директории:

```bash
./build/gigaam_asr infer sample.wav
```

Если модели лежат в отдельной папке:

```bash
./build/gigaam_asr infer sample.wav /path/to/model_dir /path/to/v3_vocab.txt
```

Где `/path/to/model_dir` содержит:

- `v3_rnnt_encoder.int8.onnx`
- `v3_rnnt_decoder.int8.onnx`
- `v3_rnnt_joint.int8.onnx`

Для совместимости старый короткий вызов тоже оставлен:

```bash
./build/gigaam_asr sample.wav
```

### 2. Прогон русского тестового набора

Бинарник умеет читать `manifest.tsv` в формате:

```text
relative/or/absolute/path.wav<TAB>reference text
```

Есть helper-скрипт для подготовки небольшого поднабора `istupakov/russian_librispeech`:

```bash
python scripts/prepare_russian_librispeech_subset.py data/ru_test_subset 10
```

После этого можно прогнать оценку:

```bash
./build/gigaam_asr eval-ru data/ru_test_subset/manifest.tsv
```

Режим `eval-ru` печатает:

- reference и hypothesis для каждого WAV;
- итоговые `WER` и `CER` по всему manifest.

## Ограничения

- decoding пока greedy, без beam search;
- пример проверялся как end-to-end smoke test, но качество на синтетическом TTS-аудио не показательно;
- для честного сравнения качества лучше прогонять реальные русские WAV, а не `espeak-ng`.
