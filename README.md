# GigaAM CLI

Нативный C++ CLI для `GigaAM v3 E2E RNN-T` через ONNX Runtime.

Текущая версия репозитория поддерживает только одну модель:

- `v3_e2e_rnnt_encoder.int8.onnx`
- `v3_e2e_rnnt_decoder.int8.onnx`
- `v3_e2e_rnnt_joint.int8.onnx`
- `v3_e2e_rnnt_vocab.txt`

Старые CTC/RNN-T без `e2e` из репозитория убраны намеренно.

## Что внутри

- единый бинарник `gigaam`
- подкоманды для скачивания моделей и публичного русского набора
- инференс одного аудиофайла
- оценка по `manifest.tsv`
- модульная архитектура: `domain`, `app`, `infra`, `cli`

## Зависимости

Нужно локально иметь:

- `cmake >= 3.16`
- C++17-компилятор
- `OpenSSL` dev package
- `curl`

Через `FetchContent` проект подтягивает:

- `ONNX Runtime`
- `CLI11`
- `cpp-httplib`
- `nlohmann/json`
- `miniaudio`

`miniaudio` используется для входных форматов:

- `wav`
- `flac`
- `mp3`
- `ogg`

## Сборка

```bash
cmake -S . -B build
cmake --build build -j
```

Если нужно использовать локальный ONNX Runtime:

```bash
cmake -S . -B build \
  -DGIGAAM_FETCH_ONNXRUNTIME=OFF \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build -j
```

### Повторить конфигурацию в Docker с `cmake 3.22`

Если нужно проверить поведение именно на `cmake 3.22`, можно запустить чистую конфигурацию в контейнере:

```bash
./scripts/repro-cmake322.sh
```

С дополнительными аргументами для `cmake`:

```bash
./scripts/repro-cmake322.sh -DGIGAAM_FETCH_ONNXRUNTIME=OFF
```

По умолчанию скрипт создаёт образ `gigaam-cmake322` и конфигурирует проект в `build-cmake322`.

## Quick Start

### У меня есть MP3 и я хочу получить транскрибацию

Если ты только что скачал репозиторий и у тебя есть файл `my_audio.mp3`, минимальный сценарий такой:

```bash
git clone <URL_ЭТОГО_РЕПОЗИТОРИЯ>
cd GigaamAsrCXX

cmake -S . -B build
cmake --build build -j

./build/gigaam models download
./build/gigaam infer my_audio.mp3 --model-dir models/gigaam-v3-e2e-rnnt
```

На выходе CLI печатает только итоговый распознанный текст.

Поддерживаются входные форматы:

- `wav`
- `flac`
- `mp3`
- `ogg`

### Я хочу быстро проверить, что всё работает

```bash
git clone <URL_ЭТОГО_РЕПОЗИТОРИЯ>
cd GigaamAsrCXX

cmake -S . -B build
cmake --build build -j

./build/gigaam models download
./build/gigaam datasets download ru-public --limit 2

./build/gigaam infer data/ru-public/audio/sample_001.wav --model-dir models/gigaam-v3-e2e-rnnt
./build/gigaam eval data/ru-public/manifest.tsv --model-dir models/gigaam-v3-e2e-rnnt --verbose
```

Ожидаемый результат:

- `infer` печатает одну строку с транскрибацией
- `eval` печатает `reference/hypothesis` и summary с `WER/CER`

## Подкоманды

### 1. Скачать модель

По умолчанию модель скачивается в `models/gigaam-v3-e2e-rnnt`:

```bash
./build/gigaam models download
```

Или в свою папку:

```bash
./build/gigaam models download --output-dir /path/to/models
```

Перекачать существующие файлы:

```bash
./build/gigaam models download --output-dir /path/to/models --force
```

### 2. Скачать публичный русский набор

Подкоманда скачивает публичные примеры из `istupakov/russian_librispeech` и готовит `manifest.tsv`.

```bash
./build/gigaam datasets download ru-public
```

Свой output dir и лимит:

```bash
./build/gigaam datasets download ru-public \
  --output-dir data/ru-public \
  --limit 10
```

### 3. Прогнать один аудиофайл

```bash
./build/gigaam infer sample.wav
```

Если модель лежит в отдельной папке:

```bash
./build/gigaam infer sample.wav --model-dir /path/to/models
```

### 4. Прогнать manifest

```bash
./build/gigaam eval data/ru-public/manifest.tsv --model-dir /path/to/models
```

Для печати `reference/hypothesis` по каждому файлу:

```bash
./build/gigaam eval data/ru-public/manifest.tsv \
  --model-dir /path/to/models \
  --verbose
```

## Полный сценарий с нуля

```bash
cmake -S . -B build
cmake --build build -j

./build/gigaam models download
./build/gigaam datasets download ru-public --limit 2

./build/gigaam infer data/ru-public/audio/sample_001.wav
./build/gigaam eval data/ru-public/manifest.tsv --model-dir models/gigaam-v3-e2e-rnnt --verbose
```

## Метрики

`eval` считает:

- `WER`
- `CER`

Перед сравнением текст нормализуется под русский кейс:

- приводится к нижнему регистру
- `ё` сводится к `е`
- пунктуация и прочие не-буквенные разделители игнорируются

Это позволяет корректно сравнивать `e2e_rnnt`, который возвращает более “человеческий” текст с регистром и пунктуацией.

## Проверенная smoke-последовательность

Было вручную прогнано:

```bash
./build-refactor/gigaam models download --output-dir /tmp/gigaam-models-e2e --force
./build-refactor/gigaam datasets download ru-public --output-dir /tmp/gigaam-ru-public --limit 2 --force
./build-refactor/gigaam infer /tmp/gigaam-ru-public/audio/sample_001.wav --model-dir /tmp/gigaam-models-e2e
./build-refactor/gigaam eval /tmp/gigaam-ru-public/manifest.tsv --model-dir /tmp/gigaam-models-e2e --verbose
```

Результат smoke test:

- `infer` вернул корректный текст для `sample_001.wav`
- `eval` на 2 публичных файлах дал `WER=0%` и `CER=0%`

## Источники

- GigaAM: https://github.com/salute-developers/GigaAM
- ONNX-конвертация: https://huggingface.co/istupakov/gigaam-v3-onnx
- Публичный русский набор: https://huggingface.co/datasets/istupakov/russian_librispeech
