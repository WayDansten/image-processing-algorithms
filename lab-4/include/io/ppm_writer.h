#pragma once

#include <string>

#include "io/image.h"

namespace pt {

struct ToneMappingSettings {
    float exposure = 1.0f;
    float gamma = 2.2f;
};

bool write_ppm(const std::string& path,
               const Image& image,
               const ToneMappingSettings& settings,
               std::string& error_message);

}
