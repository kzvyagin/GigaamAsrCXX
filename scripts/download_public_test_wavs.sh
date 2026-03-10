#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${1:-${ROOT_DIR}/data/public_test_wavs}"
LIMIT="${2:-3}"

python3 "${SCRIPT_DIR}/prepare_russian_librispeech_subset.py" "${OUT_DIR}" "${LIMIT}"

cat <<EOF

Public WAV samples downloaded to: ${OUT_DIR}

Single-file smoke test:
  ./build/gigaam_asr infer "${OUT_DIR}/audio/sample_001.wav"

Dataset evaluation:
  ./build/gigaam_asr eval-ru "${OUT_DIR}/manifest.tsv"
EOF
