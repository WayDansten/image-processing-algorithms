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

struct FaceTri {
    int v0 = -1;
    int v1 = -1;
    int v2 = -1;
    int n0 = -1;
    int n1 = -1;
    int n2 = -1;
    int material_id = -1;
    bool is_light = false;
};

bool parse_face_vertex(const std::string& token,
                       int vertex_count,
                       int normal_count,
                       int& out_vertex,
                       int& out_normal) {
    std::string v_part;
    std::string n_part;

    const auto first_slash = token.find('/');
    if (first_slash == std::string::npos) {
        v_part = token;
    } else {
        v_part = token.substr(0, first_slash);
        const auto last_slash = token.rfind('/');
        if (last_slash != std::string::npos && last_slash + 1 < token.size()) {
            n_part = token.substr(last_slash + 1);
        }
    }

    if (v_part.empty()) {
        return false;
    }

    const int raw_vertex = std::stoi(v_part);
    const int vertex_index = to_index(raw_vertex, vertex_count);
    if (vertex_index < 0 || vertex_index >= vertex_count) {
        return false;
    }

    int normal_index = -1;
    if (!n_part.empty()) {
        const int raw_normal = std::stoi(n_part);
        normal_index = to_index(raw_normal, normal_count);
        if (normal_index < 0 || normal_index >= normal_count) {
            return false;
        }
    }

    out_vertex = vertex_index;
    out_normal = normal_index;
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
    std::vector<Vec3> normals;
    std::vector<FaceTri> face_tris;
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
        } else if (keyword == "vn") {
            Vec3 n{};
            stream >> n.x >> n.y >> n.z;
            normals.push_back(n);
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
            std::vector<int> normal_indices;
            std::string token;
            while (stream >> token) {
                int vertex_index = -1;
                int normal_index = -1;
                if (!parse_face_vertex(token,
                                       static_cast<int>(vertices.size()),
                                       static_cast<int>(normals.size()),
                                       vertex_index,
                                       normal_index)) {
                    error_message = "Invalid face index in OBJ file.";
                    return false;
                }
                face_indices.push_back(vertex_index);
                normal_indices.push_back(normal_index);
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
                FaceTri tri{};
                tri.v0 = face_indices[0];
                tri.v1 = face_indices[i];
                tri.v2 = face_indices[i + 1];
                tri.n0 = normal_indices[0];
                tri.n1 = normal_indices[i];
                tri.n2 = normal_indices[i + 1];
                tri.material_id = material_id;
                tri.is_light = current_is_light && material_id >= 0;
                face_tris.push_back(tri);
            }
        }
    }

    std::vector<Vec3> smooth_normals(vertices.size(), Vec3{});
    for (const auto& tri : face_tris) {
        const Vec3& v0 = vertices[tri.v0];
        const Vec3& v1 = vertices[tri.v1];
        const Vec3& v2 = vertices[tri.v2];
        Vec3 face_normal = cross(v1 - v0, v2 - v0);
        if (length(face_normal) == 0.0f) {
            continue;
        }
        face_normal = normalize(face_normal);
        smooth_normals[tri.v0] = smooth_normals[tri.v0] + face_normal;
        smooth_normals[tri.v1] = smooth_normals[tri.v1] + face_normal;
        smooth_normals[tri.v2] = smooth_normals[tri.v2] + face_normal;
    }

    for (auto& n : smooth_normals) {
        if (length(n) > 0.0f) {
            n = normalize(n);
        }
    }

    for (const auto& tri : face_tris) {
        Triangle out{};
        out.v0 = vertices[tri.v0];
        out.v1 = vertices[tri.v1];
        out.v2 = vertices[tri.v2];
        out.material_id = tri.material_id;

        Vec3 face_normal = cross(out.v1 - out.v0, out.v2 - out.v0);
        if (length(face_normal) > 0.0f) {
            face_normal = normalize(face_normal);
        }

        out.n0 = (tri.n0 >= 0 && tri.n0 < static_cast<int>(normals.size())) ? normals[tri.n0] : smooth_normals[tri.v0];
        out.n1 = (tri.n1 >= 0 && tri.n1 < static_cast<int>(normals.size())) ? normals[tri.n1] : smooth_normals[tri.v1];
        out.n2 = (tri.n2 >= 0 && tri.n2 < static_cast<int>(normals.size())) ? normals[tri.n2] : smooth_normals[tri.v2];

        if (length(out.n0) == 0.0f) {
            out.n0 = face_normal;
        }
        if (length(out.n1) == 0.0f) {
            out.n1 = face_normal;
        }
        if (length(out.n2) == 0.0f) {
            out.n2 = face_normal;
        }

        if (length(out.n0) > 0.0f) {
            out.n0 = normalize(out.n0);
        }
        if (length(out.n1) > 0.0f) {
            out.n1 = normalize(out.n1);
        }
        if (length(out.n2) > 0.0f) {
            out.n2 = normalize(out.n2);
        }

        scene.triangles.push_back(out);

        if (tri.is_light) {
            Material& material = scene.materials[tri.material_id];
            material.is_emissive = true;
            material.emission = material.kd * emissive_scale;
            scene.light_triangle_indices.push_back(
                static_cast<int>(scene.triangles.size() - 1)
            );
        }
    }

    return true;
}

}
