#pragma once

#include "core/vec3.h"

namespace pt {

struct Camera {
    Vec3 position{0.0f, 0.0f, 0.0f};
    Vec3 forward{0.0f, 0.0f, -1.0f};
    Vec3 right{1.0f, 0.0f, 0.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float fov_y_degrees = 60.0f;
};

}
