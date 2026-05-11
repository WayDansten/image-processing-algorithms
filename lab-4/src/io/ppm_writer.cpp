#include "io/ppm_writer.h"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace pt {
namespace {

float average_intensity(const Image& image) {
    if (image.pixels.empty()) {
        return 0.0f;
    }

    float sum = 0.0f;
    for (const Vec3& pixel : image.pixels) {
        sum += max_component(pixel);
    }
    return sum / static_cast<float>(image.pixels.size());
}

unsigned char to_byte(float value) {
    const float clamped = std::clamp(value, 0.0f, 1.0f);
    return static_cast<unsigned char>(std::round(clamped * 255.0f));
}

}

bool write_ppm(const std::string& path,
               const Image& image,
               const ToneMappingSettings& settings,
               std::string& error_message) {
    if (image.size.width == 0 || image.size.height == 0) {
        error_message = "Image is empty.";
        return false;
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        error_message = "Failed to open output file: " + path;
        return false;
    }

    const float avg_value = std::max(1e-6f, average_intensity(image));
    const float exposure_scale = (settings.exposure * 0.5f) / avg_value;
    const float inv_gamma = (settings.gamma > 0.0f) ? (1.0f / settings.gamma) : 1.0f;

    file << "P6\n" << image.size.width << " " << image.size.height << "\n255\n";

    for (std::uint32_t y = 0; y < image.size.height; ++y) {
        for (std::uint32_t x = 0; x < image.size.width; ++x) {
            Vec3 color = image.at(x, y) * exposure_scale;
            color = clamp01(color);
            color.x = std::pow(color.x, inv_gamma);
            color.y = std::pow(color.y, inv_gamma);
            color.z = std::pow(color.z, inv_gamma);

            const unsigned char r = to_byte(color.x);
            const unsigned char g = to_byte(color.y);
            const unsigned char b = to_byte(color.z);
            file.write(reinterpret_cast<const char*>(&r), 1);
            file.write(reinterpret_cast<const char*>(&g), 1);
            file.write(reinterpret_cast<const char*>(&b), 1);
        }
    }

    return true;
}

}
