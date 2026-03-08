cmake_minimum_required(VERSION 3.16)
project(gigaam_asr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

option(GIGAAM_FETCH_ONNXRUNTIME "Download a prebuilt ONNX Runtime archive with FetchContent" ON)

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
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    )

    if (ONNXRUNTIME_URL_HASH)
        list(APPEND _onnxruntime_fetch_args URL_HASH "${ONNXRUNTIME_URL_HASH}")
    endif()

    FetchContent_Declare(onnxruntime_prebuilt ${_onnxruntime_fetch_args})
    FetchContent_GetProperties(onnxruntime_prebuilt)
    if (NOT onnxruntime_prebuilt_POPULATED)
        FetchContent_Populate(onnxruntime_prebuilt)
    endif()

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

add_executable(gigaam_asr main.cpp)

target_link_libraries(gigaam_asr PRIVATE
    onnxruntime
)
