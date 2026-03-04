#find_package(SDL2 REQUIRED)
#find_package(onnxruntime REQUIRED) # или указать путь вручную
# 1. Указать путь к заголовочным файлам (.h)
#include_directories(/path/to/library/include)

# 2. Указать путь к самой библиотеке (.lib / .a / .so)
#link_directories(/path/to/library/lib)

#add_executable(gigaam_asr main.cpp)
#target_link_libraries(gigaam_asr SDL2::SDL2 onnxruntime::onnxruntime)

cmake_minimum_required(VERSION 3.10)
project(gigaam_asr)

set(CMAKE_CXX_STANDARD 17)

# ONNX Runtime
#find_package(ONNXRuntime REQUIRED)  # или укажите путь вручную

# Kaldi Native Fbank
#add_subdirectory(kaldi-native-fbank)  # если склонировали репозиторий

#/home/user/cxx/kaldi-native-fbank/build/include 
#/home/user/cxx/kaldi-native-fbank/build/lib

include_directories( /home/user/cxx/kaldi-native-fbank/build/include 
                     /home/user/cxx/onnxruntime-linux-x64-1.24.2/include
                     /home/user/cxx/libsamplerate/include
                     /home/user/cxx/dr_libs )

link_directories(/home/user/cxx/kaldi-native-fbank/build/lib 
                 /home/user/cxx/onnxruntime-linux-x64-1.24.2/lib
                 /home/user/cxx/libsamplerate/src/.libs )

# libsamplerate
#find_package(SampleRate REQUIRED)

# dr_wav просто включаем в проект
add_executable(gigaam_asr main.cpp)
target_include_directories(gigaam_asr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(gigaam_asr
    onnxruntime
    kaldi-native-fbank
    samplerate
)