#pragma once

#include "core/vec3.h"

namespace pt {

struct LightSample {
    Vec3 position{};
    Vec3 normal{};
    Vec3 emission{};
    float pdf_area = 0.0f;
};

}
