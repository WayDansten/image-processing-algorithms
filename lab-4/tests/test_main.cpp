#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "accel/embree_scene.h"
#include "core/hit.h"
#include "core/ray.h"
#include "core/rng.h"
#include "core/sampling.h"
#include "core/scene.h"
#include "core/vec3.h"
#include "io/obj_loader.h"

namespace {

bool nearly_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

int test_vec3_basic() {
    pt::Vec3 a{1.0f, 2.0f, 3.0f};
    pt::Vec3 b{2.0f, 1.0f, -1.0f};
    pt::Vec3 c = a + b;
    if (!nearly_equal(c.x, 3.0f) || !nearly_equal(c.y, 3.0f) || !nearly_equal(c.z, 2.0f)) {
        return 1;
    }
    pt::Vec3 d = a - b;
    if (!nearly_equal(d.x, -1.0f) || !nearly_equal(d.y, 1.0f) || !nearly_equal(d.z, 4.0f)) {
        return 1;
    }
    if (!nearly_equal(pt::dot(a, b), 1.0f)) {
        return 1;
    }
    return 0;
}

int test_sampling_cosine() {
    pt::Rng rng(123u);
    const pt::Vec3 normal{0.0f, 1.0f, 0.0f};
    float pdf = 0.0f;
    pt::Vec3 dir = pt::sample_cosine_hemisphere(normal, rng, pdf);
    if (pt::dot(dir, normal) < 0.0f) {
        return 1;
    }
    if (pdf <= 0.0f) {
        return 1;
    }
    return 0;
}

int test_russian_roulette() {
    pt::Rng rng(1u);
    float weight = 0.0f;
    const bool survived = pt::russian_roulette(1.0f, rng, weight);
    if (!survived || !nearly_equal(weight, 1.0f)) {
        return 1;
    }
    return 0;
}

int test_embree_intersection() {
    pt::Scene scene{};
    scene.materials.push_back(pt::Material{});
    pt::Triangle tri{};
    tri.v0 = pt::Vec3{-1.0f, 0.0f, 0.0f};
    tri.v1 = pt::Vec3{1.0f, 0.0f, 0.0f};
    tri.v2 = pt::Vec3{0.0f, 1.0f, 0.0f};
    tri.material_id = 0;
    scene.triangles.push_back(tri);

    pt::EmbreeScene accel;
    std::string error_message;
    if (!accel.build(scene, error_message)) {
        return 1;
    }

    pt::Ray ray{pt::Vec3{0.0f, 0.25f, 2.0f}, pt::Vec3{0.0f, 0.0f, -1.0f}};
    pt::HitInfo hit{};
    if (!accel.intersect(ray, 0.0f, 10.0f, hit)) {
        return 1;
    }
    if (!nearly_equal(hit.position.z, 0.0f, 1e-3f)) {
        return 1;
    }
    return 0;
}

int test_obj_loader_light_group() {
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path();
    const std::filesystem::path obj_path = temp_dir / "pt_test_scene.obj";
    const std::filesystem::path mtl_path = temp_dir / "pt_test_scene.mtl";

    {
        std::ofstream mtl(mtl_path.string());
        if (!mtl) {
            return 1;
        }
        mtl << "newmtl light_mat\n";
        mtl << "Kd 1 0 0\n";
    }

    {
        std::ofstream obj(obj_path.string());
        if (!obj) {
            return 1;
        }
        obj << "mtllib " << mtl_path.filename().string() << "\n";
        obj << "g light_panel\n";
        obj << "usemtl light_mat\n";
        obj << "v 0 0 0\n";
        obj << "v 1 0 0\n";
        obj << "v 0 1 0\n";
        obj << "f 1 2 3\n";
    }

    pt::Scene scene{};
    std::string error_message;
    if (!pt::load_obj(obj_path.string(), scene, error_message)) {
        return 1;
    }

    std::error_code ec;
    std::filesystem::remove(obj_path, ec);
    std::filesystem::remove(mtl_path, ec);

    if (scene.triangles.size() != 1) {
        return 1;
    }
    if (scene.light_triangle_indices.size() != 1) {
        return 1;
    }
    if (scene.materials.empty()) {
        return 1;
    }
    if (!scene.materials[0].is_emissive) {
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_vec3_basic();
    failures += test_sampling_cosine();
    failures += test_russian_roulette();
    failures += test_embree_intersection();
    failures += test_obj_loader_light_group();

    if (failures == 0) {
        std::cout << "All tests passed." << std::endl;
        return 0;
    }

    std::cout << "Tests failed: " << failures << std::endl;
    return 1;
}
