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

    void resize(std::uint32_t width, std::uint32_t height) {
        size.width = width;
        size.height = height;
        pixels.assign(static_cast<std::size_t>(width) * height, Vec3{});
    }

    Vec3& at(std::uint32_t x, std::uint32_t y) {
        return pixels[static_cast<std::size_t>(y) * size.width + x];
    }

    const Vec3& at(std::uint32_t x, std::uint32_t y) const {
        return pixels[static_cast<std::size_t>(y) * size.width + x];
    }
};

}
