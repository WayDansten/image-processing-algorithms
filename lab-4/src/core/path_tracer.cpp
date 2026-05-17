#include "core/path_tracer.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/hit.h"
#include "core/light.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/sampling.h"

namespace pt {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInvPi = 1.0f / kPi;
constexpr float kEpsilon = 1e-4f;

struct LightDistribution {
    std::vector<int> triangle_indices;
    std::vector<float> cdf;
    float total_weight = 0.0f;
};

Vec3 triangle_normal(const Triangle& tri) {
    return normalize(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
}

float triangle_area(const Triangle& tri) {
    return 0.5f * length(cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
}

LightDistribution build_light_distribution(const Scene& scene) {
    LightDistribution distribution{};

    for (int index : scene.light_triangle_indices) {
        if (index < 0 || static_cast<std::size_t>(index) >= scene.triangles.size()) {
            continue;
        }
        const Triangle& tri = scene.triangles[static_cast<std::size_t>(index)];
        if (tri.material_id < 0 || static_cast<std::size_t>(tri.material_id) >= scene.materials.size()) {
            continue;
        }
        const Material& mat = scene.materials[static_cast<std::size_t>(tri.material_id)];
        if (!mat.is_emissive) {
            continue;
        }
        const float area = triangle_area(tri);
        const float weight = area * max_component(mat.emission);
        if (weight <= 0.0f) {
            continue;
        }
        distribution.total_weight += weight;
        distribution.triangle_indices.push_back(index);
        distribution.cdf.push_back(distribution.total_weight);
    }

    return distribution;
}

bool sample_light(const Scene& scene, const LightDistribution& distribution, Rng& rng, LightSample& out_sample) {
    if (distribution.triangle_indices.empty() || distribution.total_weight <= 0.0f) {
        return false;
    }

    const float pick = rng.uniform() * distribution.total_weight;
    auto it = std::lower_bound(distribution.cdf.begin(), distribution.cdf.end(), pick);
    const std::size_t light_index = static_cast<std::size_t>(std::distance(distribution.cdf.begin(), it));
    const int triangle_index = distribution.triangle_indices[light_index];

    const Triangle& tri = scene.triangles[static_cast<std::size_t>(triangle_index)];
    const Material& mat = scene.materials[static_cast<std::size_t>(tri.material_id)];
    const float area = triangle_area(tri);

    const float r1 = std::sqrt(rng.uniform());
    const float r2 = rng.uniform();
    const float a = 1.0f - r1;
    const float b = r1 * (1.0f - r2);
    const float c = r1 * r2;

    out_sample.position = tri.v0 * a + tri.v1 * b + tri.v2 * c;
    out_sample.normal = triangle_normal(tri);
    out_sample.emission = mat.emission;

    const float select_pdf = (distribution.total_weight > 0.0f)
        ? (distribution.cdf[light_index] - (light_index == 0 ? 0.0f : distribution.cdf[light_index - 1])) / distribution.total_weight
        : 0.0f;
    out_sample.pdf_area = (area > 0.0f && select_pdf > 0.0f) ? (select_pdf / area) : 0.0f;
    return out_sample.pdf_area > 0.0f;
}

Ray generate_camera_ray(const Camera& camera,
                        std::uint32_t x,
                        std::uint32_t y,
                        const RenderSettings& settings,
                        Rng& rng) {
    const float u = (static_cast<float>(x) + rng.uniform()) / static_cast<float>(settings.width);
    const float v = (static_cast<float>(y) + rng.uniform()) / static_cast<float>(settings.height);

    const float aspect = static_cast<float>(settings.width) / static_cast<float>(settings.height);
    const float scale = std::tan(0.5f * camera.fov_y_degrees * kPi / 180.0f);
    const float px = (2.0f * u - 1.0f) * aspect * scale;
    const float py = (1.0f - 2.0f * v) * scale;

    Vec3 direction = normalize(camera.forward + camera.right * px + camera.up * py);
    return Ray{camera.position, direction};
}

}

bool PathTracer::render(const Scene& scene,
                        const Camera& camera,
                        const RenderSettings& settings,
                        Image& out_image,
                        std::string& error_message) const {
    EmbreeScene accel;
    if (!accel.build(scene, error_message)) {
        return false;
    }

    out_image.resize(settings.width, settings.height);
    LightDistribution lights = build_light_distribution(scene);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < static_cast<int>(settings.height); ++y) {
        for (std::uint32_t x = 0; x < settings.width; ++x) {
            const std::uint32_t pixel_index = static_cast<std::uint32_t>(y) * settings.width + x;
            Rng rng(settings.seed + pixel_index * 9781u + 1u);
            Vec3 pixel_radiance{};

            for (std::uint32_t s = 0; s < settings.samples_per_pixel; ++s) {
                Ray ray = generate_camera_ray(camera, x, y, settings, rng);
                Vec3 throughput{1.0f, 1.0f, 1.0f};
                Vec3 radiance{};
                bool last_bounce_specular = true;

                for (std::uint32_t depth = 0; depth < settings.max_depth; ++depth) {
                    HitInfo hit{};
                    if (!accel.intersect(ray, kEpsilon, 1e30f, hit)) {
                        break;
                    }

                    const Material& material = scene.materials[static_cast<std::size_t>(hit.material_id)];
                    if (material.is_emissive) {
                        if (last_bounce_specular) {
                            radiance += throughput * material.emission;
                        }
                        break;
                    }

                    LightSample light_sample{};
                    if (sample_light(scene, lights, rng, light_sample)) {
                        const Vec3 to_light = light_sample.position - hit.position;
                        const float distance_sq = dot(to_light, to_light);
                        const float distance = std::sqrt(distance_sq);
                        const Vec3 dir_to_light = to_light / distance;

                        const float cos_surface = std::max(0.0f, dot(hit.normal, dir_to_light));
                        Vec3 light_normal = light_sample.normal;
                        if (dot(light_normal, -dir_to_light) < 0.0f) {
                            light_normal = -light_normal;
                        }
                        const float cos_light = std::max(0.0f, dot(light_normal, -dir_to_light));

                        if (cos_surface > 0.0f && cos_light > 0.0f) {
                            const Vec3 offset_normal = (length(hit.geom_normal) > 0.0f) ? hit.geom_normal : hit.normal;
                            Ray shadow_ray{hit.position + offset_normal * kEpsilon, dir_to_light};
                            HitInfo shadow_hit{};
                            const bool occluded = accel.intersect(shadow_ray, kEpsilon, distance - kEpsilon, shadow_hit);

                            if (!occluded) {
                                const float geometry = (cos_surface * cos_light) / distance_sq;
                                const Vec3 brdf = material.kd * kInvPi;
                                const float weight = geometry / light_sample.pdf_area;

                                Vec3 contrib = throughput * brdf * light_sample.emission * weight;
                                contrib = clamp_max(contrib, 10.0f);
                                radiance += contrib;
                            }
                        }
                    }

                    const float kd_weight = max_component(material.kd);
                    const float ks_weight = max_component(material.ks);
                    const float sum = kd_weight + ks_weight;
                    if (sum <= 0.0f) {
                        break;
                    }

                    const float diffuse_prob = kd_weight / sum;
                    if (rng.uniform() < diffuse_prob) {
                        float pdf = 0.0f;
                        const Vec3 new_dir = sample_cosine_hemisphere(hit.normal, rng, pdf);
                        const Vec3 offset_normal = (length(hit.geom_normal) > 0.0f) ? hit.geom_normal : hit.normal;
                        ray = Ray{hit.position + offset_normal * kEpsilon, new_dir};
                        throughput = throughput * material.kd;
                        last_bounce_specular = false;
                    } else {
                        const Vec3 new_dir = reflect_direction(ray.direction, hit.normal);
                        const Vec3 offset_normal = (length(hit.geom_normal) > 0.0f) ? hit.geom_normal : hit.normal;
                        ray = Ray{hit.position + offset_normal * kEpsilon, new_dir};
                        throughput = throughput * material.ks;
                        last_bounce_specular = true;
                    }

                    if (depth >= 3) {
                        const float continue_prob = std::max(settings.russian_roulette_threshold, max_component(throughput));
                        float weight = 0.0f;
                        if (!russian_roulette(continue_prob, rng, weight)) {
                            break;
                        }
                        throughput = throughput * weight;
                    }
                }

                pixel_radiance += radiance;
            }

            out_image.at(x, y) = pixel_radiance / static_cast<float>(settings.samples_per_pixel);
        }
    }

    return true;
}

}
