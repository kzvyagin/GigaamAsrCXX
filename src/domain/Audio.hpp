#pragma once

#include <vector>

namespace gigaam::domain {

struct AudioBuffer {
    int sample_rate = 0;
    std::vector<float> samples;
};

}  // namespace gigaam::domain
