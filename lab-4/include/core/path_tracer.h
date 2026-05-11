#pragma once

#include <string>

#include "core/camera.h"
#include "core/scene.h"
#include "io/image.h"
#include "accel/embree_scene.h"

namespace pt {

struct RenderSettings {
    std::uint32_t width = 128;
    std::uint32_t height = 128;
    std::uint32_t samples_per_pixel = 8;
    std::uint32_t max_depth = 6;
    std::uint32_t seed = 1337;
    float russian_roulette_threshold = 0.1f;
};

class PathTracer {
public:
    bool render(const Scene& scene,
                const Camera& camera,
                const RenderSettings& settings,
                Image& out_image,
                std::string& error_message) const;
};

}
