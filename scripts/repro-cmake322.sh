#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${IMAGE_NAME:-gigaam-cmake322}"
build_dir="${BUILD_DIR:-build-cmake322}"

echo "[1/3] Building Docker image ${image_name}"
docker build -f "${repo_root}/Dockerfile.cmake322" -t "${image_name}" "${repo_root}"

echo "[2/3] Checking CMake version inside container"
docker run --rm "${image_name}" cmake --version

echo "[3/3] Running clean configure into ${build_dir}"
rm -rf "${repo_root}/${build_dir}"
docker run --rm -v "${repo_root}:/src" -w /src "${image_name}" \
    cmake -S . -B "${build_dir}" "$@"
