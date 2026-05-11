#pragma once

#include <cstdint>
#include <random>

namespace pt {

class Rng {
public:
    explicit Rng(std::uint32_t seed) : engine_(seed), dist_(0.0f, 1.0f) {}

    float uniform() { return dist_(engine_); }

private:
    std::mt19937 engine_;
    std::uniform_real_distribution<float> dist_;
};

}
