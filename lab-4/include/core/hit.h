#pragma once

#include "core/vec3.h"

namespace pt {

struct HitInfo {
    bool hit = false;
    float t = 0.0f;
    Vec3 position{};
    Vec3 normal{};
    Vec3 geom_normal{};
    int material_id = -1;
    int object_id = -1;
};

}
