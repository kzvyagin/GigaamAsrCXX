cmake_minimum_required(VERSION 3.16)
project(gigaam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
include(CTest)

option(GIGAAM_FETCH_ONNXRUNTIME "Download a prebuilt ONNX Runtime archive with FetchContent" ON)

set(GIGAAM_FETCHCONTENT_ARGS)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    list(APPEND GIGAAM_FETCHCONTENT_ARGS DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
endif()

set(ONNXRUNTIME_VERSION "1.18.1" CACHE STRING "ONNX Runtime version to download when FetchContent is enabled")
set(ONNXRUNTIME_URL "" CACHE STRING "Explicit URL to a prebuilt ONNX Runtime archive")
set(ONNXRUNTIME_URL_HASH "" CACHE STRING "Optional URL_HASH for the ONNX Runtime archive")
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Path to ONNX Runtime installation")

if (GIGAAM_FETCH_ONNXRUNTIME)
    if (NOT ONNXRUNTIME_URL)
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
            set(_onnxruntime_archive "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
            set(_onnxruntime_archive "onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}.tgz")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
            set(_onnxruntime_archive "onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz")
        else()
            message(FATAL_ERROR
                "Cannot derive ONNX Runtime archive name for platform "
                "${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}. "
                "Set ONNXRUNTIME_URL explicitly or disable GIGAAM_FETCH_ONNXRUNTIME.")
        endif()

        set(ONNXRUNTIME_URL
            "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${_onnxruntime_archive}"
            CACHE STRING "Explicit URL to a prebuilt ONNX Runtime archive" FORCE)
    endif()

    set(_onnxruntime_fetch_args
        URL "${ONNXRUNTIME_URL}"
    )
    list(APPEND _onnxruntime_fetch_args ${GIGAAM_FETCHCONTENT_ARGS})

    if (ONNXRUNTIME_URL_HASH)
        list(APPEND _onnxruntime_fetch_args URL_HASH "${ONNXRUNTIME_URL_HASH}")
    endif()

    FetchContent_Declare(onnxruntime_prebuilt ${_onnxruntime_fetch_args})
    FetchContent_MakeAvailable(onnxruntime_prebuilt)
    set(ONNXRUNTIME_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}" CACHE PATH "Path to ONNX Runtime installation" FORCE)
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES
        include
        include/onnxruntime
        include/onnxruntime/core/session
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES
        lib
        lib64
)

if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime not found. Set ONNXRUNTIME_ROOT to the extracted ONNX Runtime directory.\n"
        "Expected layout: <root>/include and <root>/lib/libonnxruntime.so")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

FetchContent_Declare(
    CLI11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.tar.gz
    ${GIGAAM_FETCHCONTENT_ARGS}
)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
    cpphttplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.15.3.tar.gz
    ${GIGAAM_FETCHCONTENT_ARGS}
)
FetchContent_MakeAvailable(cpphttplib)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    ${GIGAAM_FETCHCONTENT_ARGS}
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    miniaudio_src
    URL https://github.com/mackron/miniaudio/archive/refs/tags/0.11.21.tar.gz
    ${GIGAAM_FETCHCONTENT_ARGS}
)
FetchContent_MakeAvailable(miniaudio_src)

find_package(OpenSSL REQUIRED)

add_library(miniaudio INTERFACE)
target_include_directories(miniaudio INTERFACE "${miniaudio_src_SOURCE_DIR}")

add_library(cpphttplib INTERFACE)
target_include_directories(cpphttplib INTERFACE "${cpphttplib_SOURCE_DIR}")
target_compile_definitions(cpphttplib INTERFACE CPPHTTPLIB_OPENSSL_SUPPORT)
target_link_libraries(cpphttplib INTERFACE OpenSSL::SSL OpenSSL::Crypto)

set(GIGAAM_SOURCES
    src/app/Ports.hpp
    src/app/Services.cpp
    src/app/Services.hpp
    src/cli/main.cpp
    src/domain/Assets.cpp
    src/domain/Assets.hpp
    src/domain/Audio.hpp
    src/domain/Dataset.cpp
    src/domain/Dataset.hpp
    src/domain/Manifest.cpp
    src/domain/Manifest.hpp
    src/domain/Text.cpp
    src/domain/Text.hpp
    src/infra/asr/RnntRecognizer.cpp
    src/infra/asr/RnntRecognizer.hpp
    src/infra/audio/MiniaudioDecoder.cpp
    src/infra/audio/MiniaudioDecoder.hpp
    src/infra/datasets/RussianLibrispeechIndex.cpp
    src/infra/datasets/RussianLibrispeechIndex.hpp
    src/infra/download/HttplibHttpClient.cpp
    src/infra/download/HttplibHttpClient.hpp
)

add_executable(gigaam ${GIGAAM_SOURCES})
target_include_directories(gigaam PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(gigaam PRIVATE
    CLI11::CLI11
    cpphttplib
    miniaudio
    nlohmann_json::nlohmann_json
    onnxruntime
)

if (BUILD_TESTING)
    add_executable(gigaam_domain_tests
        tests/DomainTests.cpp
        src/domain/Assets.cpp
        src/domain/Manifest.cpp
        src/domain/Text.cpp
    )
    target_include_directories(gigaam_domain_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    add_test(NAME gigaam_domain_tests COMMAND gigaam_domain_tests)
endif()
