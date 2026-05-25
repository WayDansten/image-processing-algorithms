#pragma once

#include <string>

#include "io/image.h"

namespace pt {

struct BilateralFilterSettings {
    int radius = 5;
    float sigma_spatial = 2.0f;
    float sigma_depth = 0.1f;
    float sigma_normal = 0.2f;
};

bool apply_bilateral_filter(const Image& input,
                            Image& output,
                            const BilateralFilterSettings& settings,
                            std::string& error_message);

}
