#pragma once

#include <vector>

#include "core/material.h"
#include "core/vec3.h"

namespace pt {

struct Triangle {
    Vec3 v0{};
    Vec3 v1{};
    Vec3 v2{};
    int material_id = -1;
};

struct Scene {
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<int> light_triangle_indices;
};

}
