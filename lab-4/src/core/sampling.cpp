#include "core/sampling.h"

#include <algorithm>
#include <cmath>

namespace pt {
namespace {

constexpr float kPi = 3.14159265358979323846f;

} // namespace

OrthonormalBasis build_onb(const Vec3& normal) {
    OrthonormalBasis basis{};
    basis.w = normalize(normal);

    const Vec3 a = (std::abs(basis.w.z) < 0.999f) ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
    basis.u = normalize(cross(a, basis.w));
    basis.v = cross(basis.w, basis.u);
    return basis;
}

Vec3 sample_cosine_hemisphere(const Vec3& normal, Rng& rng, float& out_pdf) {
    const float e_phi = rng.uniform();
    const float e_theta = rng.uniform();

    const float phi = 2.0f * kPi * e_phi;
    const float sin_theta = std::sqrt(e_theta);
    const float cos_theta = std::sqrt(1.0f - e_theta);

    const OrthonormalBasis basis = build_onb(normal);
    const Vec3 local = basis.u * (std::cos(phi) * sin_theta)
        + basis.v * (std::sin(phi) * sin_theta)
        + basis.w * cos_theta;

    out_pdf = cos_theta / kPi;
    return normalize(local);
}

Vec3 reflect_direction(const Vec3& incident, const Vec3& normal) {
    return normalize(reflect(incident, normal));
}

bool russian_roulette(float continue_probability, Rng& rng, float& out_weight) {
    const float clamped = std::clamp(continue_probability, 0.0f, 1.0f);
    if (clamped <= 0.0f) {
        out_weight = 0.0f;
        return false;
    }

    if (rng.uniform() < clamped) {
        out_weight = 1.0f / clamped;
        return true;
    }

    out_weight = 0.0f;
    return false;
}

}
