#pragma once

#include "core/vec3.h"

namespace pt {

struct HitInfo {
    bool hit = false;
    float t = 0.0f;
    Vec3 position{};
    Vec3 normal{};
    int material_id = -1;
};

}
