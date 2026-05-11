#pragma once

#include "core/ray.h"
#include "core/rng.h"
#include "core/vec3.h"

namespace pt {

struct OrthonormalBasis {
    Vec3 u{};
    Vec3 v{};
    Vec3 w{};
};

OrthonormalBasis build_onb(const Vec3& normal);

Vec3 sample_cosine_hemisphere(const Vec3& normal, Rng& rng, float& out_pdf);

Vec3 reflect_direction(const Vec3& incident, const Vec3& normal);

bool russian_roulette(float continue_probability, Rng& rng, float& out_weight);

}
