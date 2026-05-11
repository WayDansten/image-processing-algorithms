#pragma once

#include "core/vec3.h"

namespace pt {

struct Material {
    Vec3 kd{0.8f, 0.8f, 0.8f};
    Vec3 ks{0.0f, 0.0f, 0.0f};
    bool is_emissive = false;
    Vec3 emission{0.0f, 0.0f, 0.0f};
};

}
