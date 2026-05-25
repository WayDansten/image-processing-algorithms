#include "io/bilateral_filter.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace pt {
namespace {

struct OffsetWeight {
    int dx = 0;
    int dy = 0;
    float weight = 0.0f;
};

float gaussian_weight(float value, float sigma) {
    const float safe_sigma = std::max(sigma, 1e-6f);
    const float denom = 2.0f * safe_sigma * safe_sigma;
    return std::exp(-(value * value) / denom);
}

float normal_weight(const Vec3& a, const Vec3& b, float sigma_normal) {
    const float len_a = length(a);
    const float len_b = length(b);
    if (len_a == 0.0f && len_b == 0.0f) {
        return 1.0f;
    }
    if (len_a == 0.0f || len_b == 0.0f) {
        return 0.0f;
    }
    const Vec3 na = a / len_a;
    const Vec3 nb = b / len_b;
    const float ndot = std::clamp(dot(na, nb), -1.0f, 1.0f);
    const float diff = 1.0f - ndot;
    return gaussian_weight(diff, sigma_normal);
}

}

bool apply_bilateral_filter(const Image& input,
                            Image& output,
                            const BilateralFilterSettings& settings,
                            std::string& error_message) {
    if (input.size.width == 0 || input.size.height == 0) {
        error_message = "Image is empty.";
        return false;
    }

    const std::size_t pixel_count = input.pixels.size();
    if (input.depth.size() != pixel_count ||
        input.normals.size() != pixel_count ||
        input.object_ids.size() != pixel_count) {
        error_message = "G-buffer sizes do not match image size.";
        return false;
    }

    const int radius = std::max(settings.radius, 0);
    if (radius == 0) {
        output = input;
        return true;
    }

    const float sigma_spatial = std::max(settings.sigma_spatial, 1e-6f);
    const float sigma_depth = std::max(settings.sigma_depth, 1e-6f);
    const float sigma_normal = std::max(settings.sigma_normal, 1e-6f);

    std::vector<OffsetWeight> offsets;
    offsets.reserve(static_cast<std::size_t>((radius * 2 + 1) * (radius * 2 + 1)));

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const float distance = std::sqrt(static_cast<float>(dx * dx + dy * dy));
            const float weight = gaussian_weight(distance, sigma_spatial);
            offsets.push_back({dx, dy, weight});
        }
    }

    output.resize(input.size.width, input.size.height);
    output.depth = input.depth;
    output.normals = input.normals;
    output.object_ids = input.object_ids;

    for (std::uint32_t y = 0; y < input.size.height; ++y) {
        for (std::uint32_t x = 0; x < input.size.width; ++x) {
            const Vec3 base_color = input.at(x, y);
            const float base_depth = input.depth_at(x, y);
            const Vec3 base_normal = input.normal_at(x, y);
            const int base_object = input.object_id_at(x, y);

            Vec3 sum{};
            float weight_sum = 0.0f;

            for (const auto& offset : offsets) {
                const int nx = static_cast<int>(x) + offset.dx;
                const int ny = static_cast<int>(y) + offset.dy;
                if (nx < 0 || ny < 0 ||
                    nx >= static_cast<int>(input.size.width) ||
                    ny >= static_cast<int>(input.size.height)) {
                    continue;
                }

                const std::uint32_t ux = static_cast<std::uint32_t>(nx);
                const std::uint32_t uy = static_cast<std::uint32_t>(ny);
                const int neighbor_object = input.object_id_at(ux, uy);
                if (neighbor_object != base_object) {
                    continue;
                }

                const float depth_q = input.depth_at(ux, uy);
                const Vec3 normal_q = input.normal_at(ux, uy);

                const float w_depth = gaussian_weight(base_depth - depth_q, sigma_depth);
                const float w_normal = normal_weight(base_normal, normal_q, sigma_normal);
                const float weight = offset.weight * w_depth * w_normal;

                if (weight > 0.0f) {
                    sum += input.at(ux, uy) * weight;
                    weight_sum += weight;
                }
            }

            if (weight_sum > 0.0f) {
                output.at(x, y) = sum / weight_sum;
            } else {
                output.at(x, y) = base_color;
            }
        }
    }

    return true;
}

}
