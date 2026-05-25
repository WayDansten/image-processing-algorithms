#pragma once

#include <cstdint>
#include <vector>

#include "core/vec3.h"

namespace pt {

struct ImageSize {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct Image {
    ImageSize size{};
    std::vector<Vec3> pixels;
    std::vector<float> depth;
    std::vector<Vec3> normals;
    std::vector<int> object_ids;

    void resize(std::uint32_t width, std::uint32_t height) {
        size.width = width;
        size.height = height;
        const std::size_t count = static_cast<std::size_t>(width) * height;
        pixels.assign(count, Vec3{});
        depth.assign(count, 0.0f);
        normals.assign(count, Vec3{});
        object_ids.assign(count, -1);
    }

    Vec3& at(std::uint32_t x, std::uint32_t y) {
        return pixels[static_cast<std::size_t>(y) * size.width + x];
    }

    const Vec3& at(std::uint32_t x, std::uint32_t y) const {
        return pixels[static_cast<std::size_t>(y) * size.width + x];
    }

    float& depth_at(std::uint32_t x, std::uint32_t y) {
        return depth[static_cast<std::size_t>(y) * size.width + x];
    }

    const float& depth_at(std::uint32_t x, std::uint32_t y) const {
        return depth[static_cast<std::size_t>(y) * size.width + x];
    }

    Vec3& normal_at(std::uint32_t x, std::uint32_t y) {
        return normals[static_cast<std::size_t>(y) * size.width + x];
    }

    const Vec3& normal_at(std::uint32_t x, std::uint32_t y) const {
        return normals[static_cast<std::size_t>(y) * size.width + x];
    }

    int& object_id_at(std::uint32_t x, std::uint32_t y) {
        return object_ids[static_cast<std::size_t>(y) * size.width + x];
    }

    const int& object_id_at(std::uint32_t x, std::uint32_t y) const {
        return object_ids[static_cast<std::size_t>(y) * size.width + x];
    }
};

}
