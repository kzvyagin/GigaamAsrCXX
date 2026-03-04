Простой пример для Linux демонстрирующее работу с GigaamAsr на языке С++ 
Загрузка модели GigaamAsr происходит через библиотеку onnxruntime 

Дополнительные зависимости 
| Name  |  Link | Role |
|-------|-----|------------|
| kaldi-native-fbank | https://github.com/csukuangfj/kaldi-native-fbank  | convert to features  |
| libsamplerate    | https://github.com/libsndfile/libsamplerate  | changing samplerate   |
| dr_libs  | https://github.com/mackron/dr_libs | Audio decoding libraries for C/C++ |
| onnxruntime  | https://github.com/microsoft/onnxruntime/releases  | run inference |
 

  использованы материалы на python https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-punct-giga-am-v3-russian-2025-12-16/blob/main/test-onnx-ctc.py


 