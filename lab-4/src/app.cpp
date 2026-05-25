#include "app.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "core/path_tracer.h"
#include "io/bilateral_filter.h"
#include "io/obj_loader.h"
#include "io/ppm_writer.h"

namespace pt {

namespace {

bool parse_uint32(const std::string& value, std::uint32_t& out) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    out = static_cast<std::uint32_t>(parsed);
    return true;
}

bool parse_float(const std::string& value, float& out) {
    char* end = nullptr;
    const float parsed = std::strtof(value.c_str(), &end);
    if (!end || *end != '\0') {
        return false;
    }
    out = parsed;
    return true;
}

void print_usage() {
    std::cout << "Usage: path_tracer [options]\n"
              << "  --scene <path>       OBJ file path\n"
              << "  --out <path>         output PPM path\n"
              << "  --width <value>      image width\n"
              << "  --height <value>     image height\n"
              << "  --spp <value>        samples per pixel\n"
              << "  --max-depth <value>  maximum path depth\n"
              << "  --seed <value>       RNG seed\n"
              << "  --cam-pos x y z      camera position\n"
              << "  --cam-look x y z     look-at target\n"
              << "  --exposure <value>   exposure scale\n"
              << "  --gamma <value>      gamma correction\n";
}

}

int App::run(int argc, char** argv) {
    std::string scene_path = "scene_simple.obj";
    std::string output_path = "output.ppm";
    Vec3 camera_pos{0.0f, 1.2f, 3.5f};
    Vec3 camera_look{0.0f, 1.1f, 0.0f};

    RenderSettings settings{};
    ToneMappingSettings tone_map{};

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](std::string& out) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << std::endl;
                return false;
            }
            out = argv[++i];
            return true;
        };

        auto require_vec3 = [&](Vec3& out) -> bool {
            if (i + 3 >= argc) {
                std::cerr << "Missing vector value for " << arg << std::endl;
                return false;
            }
            std::string x_str = argv[++i];
            std::string y_str = argv[++i];
            std::string z_str = argv[++i];
            if (!parse_float(x_str, out.x) || !parse_float(y_str, out.y) || !parse_float(z_str, out.z)) {
                std::cerr << "Invalid vector value for " << arg << std::endl;
                return false;
            }
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--scene") {
            if (!require_value(scene_path)) {
                return 1;
            }
        } else if (arg == "--out") {
            if (!require_value(output_path)) {
                return 1;
            }
        } else if (arg == "--width") {
            std::string value;
            if (!require_value(value) || !parse_uint32(value, settings.width)) {
                std::cerr << "Invalid width value." << std::endl;
                return 1;
            }
        } else if (arg == "--height") {
            std::string value;
            if (!require_value(value) || !parse_uint32(value, settings.height)) {
                std::cerr << "Invalid height value." << std::endl;
                return 1;
            }
        } else if (arg == "--spp") {
            std::string value;
            if (!require_value(value) || !parse_uint32(value, settings.samples_per_pixel)) {
                std::cerr << "Invalid spp value." << std::endl;
                return 1;
            }
        } else if (arg == "--max-depth") {
            std::string value;
            if (!require_value(value) || !parse_uint32(value, settings.max_depth)) {
                std::cerr << "Invalid max depth value." << std::endl;
                return 1;
            }
        } else if (arg == "--seed") {
            std::string value;
            if (!require_value(value) || !parse_uint32(value, settings.seed)) {
                std::cerr << "Invalid seed value." << std::endl;
                return 1;
            }
        } else if (arg == "--cam-pos") {
            if (!require_vec3(camera_pos)) {
                return 1;
            }
        } else if (arg == "--cam-look") {
            if (!require_vec3(camera_look)) {
                return 1;
            }
        } else if (arg == "--exposure") {
            std::string value;
            if (!require_value(value) || !parse_float(value, tone_map.exposure)) {
                std::cerr << "Invalid exposure value." << std::endl;
                return 1;
            }
        } else if (arg == "--gamma") {
            std::string value;
            if (!require_value(value) || !parse_float(value, tone_map.gamma)) {
                std::cerr << "Invalid gamma value." << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }

    std::filesystem::path out_path(output_path);
    if (!out_path.has_parent_path()) {
        out_path = std::filesystem::path("lab-4") / "output" / out_path;
    }
    std::filesystem::create_directories(out_path.parent_path());
    output_path = out_path.string();

    Scene scene{};
    std::string error_message;
    if (!load_obj(scene_path, scene, error_message)) {
        std::cerr << "Failed to load scene: " << error_message << std::endl;
        return 1;
    }

    Camera camera{};
    camera.position = camera_pos;
    camera.forward = normalize(camera_look - camera_pos);
    camera.right = normalize(cross(camera.forward, Vec3{0.0f, 1.0f, 0.0f}));
    camera.up = normalize(cross(camera.right, camera.forward));

    Image image{};
    PathTracer tracer;
    if (!tracer.render(scene, camera, settings, image, error_message)) {
        std::cerr << "Render failed: " << error_message << std::endl;
        return 1;
    }

    Image filtered{};
    BilateralFilterSettings filter_settings{};
    if (!apply_bilateral_filter(image, filtered, filter_settings, error_message)) {
        std::cerr << "Filter failed: " << error_message << std::endl;
        return 1;
    }

    if (!write_ppm(output_path, filtered, tone_map, error_message)) {
        std::cerr << "Failed to write output: " << error_message << std::endl;
        return 1;
    }

    std::cout << "Rendered " << settings.width << "x" << settings.height
              << " to " << output_path << std::endl;
    return 0;
}

}
