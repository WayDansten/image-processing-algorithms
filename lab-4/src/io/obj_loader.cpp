#include "io/obj_loader.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace pt {
namespace {

std::string trim(const std::string& text) {
    const auto start = text.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(start, end - start + 1);
}

bool starts_with(const std::string& text, const std::string& prefix) {
    if (text.size() < prefix.size()) {
        return false;
    }
    return text.compare(0, prefix.size(), prefix) == 0;
}

int to_index(int obj_index, int count) {
    if (obj_index > 0) {
        return obj_index - 1;
    }
    if (obj_index < 0) {
        return count + obj_index;
    }
    return -1;
}

bool parse_face_vertex(const std::string& token, int vertex_count, int& out_index) {
    std::string first;
    const auto slash = token.find('/');
    if (slash == std::string::npos) {
        first = token;
    } else {
        first = token.substr(0, slash);
    }

    if (first.empty()) {
        return false;
    }

    const int raw_index = std::stoi(first);
    const int index = to_index(raw_index, vertex_count);
    if (index < 0 || index >= vertex_count) {
        return false;
    }

    out_index = index;
    return true;
}

bool load_mtl(const std::filesystem::path& path,
              std::unordered_map<std::string, int>& material_ids,
              std::vector<Material>& materials,
              std::string& error_message) {
    std::ifstream file(path);
    if (!file) {
        error_message = "Failed to open MTL file: " + path.string();
        return false;
    }

    std::string line;
    std::string current_name;
    Material current_material{};
    bool has_current = false;

    auto commit_material = [&]() {
        if (!has_current) {
            return;
        }
        const auto existing = material_ids.find(current_name);
        if (existing == material_ids.end()) {
            const int id = static_cast<int>(materials.size());
            materials.push_back(current_material);
            material_ids[current_name] = id;
        } else {
            materials[existing->second] = current_material;
        }
    };

    while (std::getline(file, line)) {
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        std::istringstream stream(trimmed);
        std::string keyword;
        stream >> keyword;
        if (keyword == "newmtl") {
            commit_material();
            stream >> current_name;
            current_material = Material{};
            has_current = !current_name.empty();
        } else if (keyword == "Kd") {
            stream >> current_material.kd.x >> current_material.kd.y >> current_material.kd.z;
        } else if (keyword == "Ks") {
            stream >> current_material.ks.x >> current_material.ks.y >> current_material.ks.z;
        }
    }

    commit_material();
    return true;
}

}

bool load_obj(const std::string& path, Scene& scene, std::string& error_message) {
    std::ifstream file(path);
    if (!file) {
        error_message = "Failed to open OBJ file: " + path;
        return false;
    }

    std::filesystem::path obj_path(path);
    const auto base_dir = obj_path.parent_path();

    std::vector<Vec3> vertices;
    std::unordered_map<std::string, int> material_ids;

    const float emissive_scale = 8.0f;

    std::string current_material_name;
    bool current_is_light = false;

    std::string line;
    while (std::getline(file, line)) {
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        std::istringstream stream(trimmed);
        std::string keyword;
        stream >> keyword;

        if (keyword == "mtllib") {
            std::string mtl_name;
            stream >> mtl_name;
            if (!mtl_name.empty()) {
                std::filesystem::path mtl_path = base_dir / mtl_name;
                if (!load_mtl(mtl_path, material_ids, scene.materials, error_message)) {
                    return false;
                }
            }
        } else if (keyword == "v") {
            Vec3 v{};
            stream >> v.x >> v.y >> v.z;
            vertices.push_back(v);
        } else if (keyword == "g") {
            current_is_light = false;
            std::string group_name;
            while (stream >> group_name) {
                if (starts_with(group_name, "light_")) {
                    current_is_light = true;
                }
            }
        } else if (keyword == "usemtl") {
            stream >> current_material_name;
        } else if (keyword == "f") {
            std::vector<int> face_indices;
            std::string token;
            while (stream >> token) {
                int index = -1;
                if (!parse_face_vertex(token, static_cast<int>(vertices.size()), index)) {
                    error_message = "Invalid face index in OBJ file.";
                    return false;
                }
                face_indices.push_back(index);
            }

            if (face_indices.size() < 3) {
                continue;
            }

            int material_id = -1;
            if (!current_material_name.empty()) {
                const auto existing = material_ids.find(current_material_name);
                if (existing == material_ids.end()) {
                    material_id = static_cast<int>(scene.materials.size());
                    scene.materials.push_back(Material{});
                    material_ids[current_material_name] = material_id;
                } else {
                    material_id = existing->second;
                }
            }

            for (std::size_t i = 1; i + 1 < face_indices.size(); ++i) {
                Triangle tri{};
                tri.v0 = vertices[face_indices[0]];
                tri.v1 = vertices[face_indices[i]];
                tri.v2 = vertices[face_indices[i + 1]];
                tri.material_id = material_id;
                scene.triangles.push_back(tri);

                if (current_is_light && material_id >= 0) {
                    Material& material = scene.materials[material_id];
                    material.is_emissive = true;
                    material.emission = material.kd * emissive_scale;
                    scene.light_triangle_indices.push_back(
                        static_cast<int>(scene.triangles.size() - 1)
                    );
                }
            }
        }
    }

    return true;
}

}
