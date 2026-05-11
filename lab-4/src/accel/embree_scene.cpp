#include "accel/embree_scene.h"

#include <embree4/rtcore.h>


namespace pt {

EmbreeScene::EmbreeScene() {
    device_ = rtcNewDevice(nullptr);
}

EmbreeScene::~EmbreeScene() {
    if (rtc_scene_) {
        rtcReleaseScene(static_cast<RTCScene>(rtc_scene_));
        rtc_scene_ = nullptr;
    }
    if (device_) {
        rtcReleaseDevice(static_cast<RTCDevice>(device_));
        device_ = nullptr;
    }
}

bool EmbreeScene::build(const Scene& scene, std::string& error_message) {
    if (!device_) {
        error_message = "Embree device is not initialized.";
        return false;
    }

    if (rtc_scene_) {
        rtcReleaseScene(static_cast<RTCScene>(rtc_scene_));
        rtc_scene_ = nullptr;
    }

    scene_ = &scene;
    rtc_scene_ = rtcNewScene(static_cast<RTCDevice>(device_));

    const std::size_t triangle_count = scene.triangles.size();
    if (triangle_count == 0) {
        return true;
    }

    RTCGeometry geometry = rtcNewGeometry(static_cast<RTCDevice>(device_), RTC_GEOMETRY_TYPE_TRIANGLE);

    const std::size_t vertex_count = triangle_count * 3;
    float* vertices = static_cast<float*>(rtcSetNewGeometryBuffer(
        geometry,
        RTC_BUFFER_TYPE_VERTEX,
        0,
        RTC_FORMAT_FLOAT3,
        sizeof(float) * 3,
        vertex_count
    ));

    unsigned* indices = static_cast<unsigned*>(rtcSetNewGeometryBuffer(
        geometry,
        RTC_BUFFER_TYPE_INDEX,
        0,
        RTC_FORMAT_UINT3,
        sizeof(unsigned) * 3,
        triangle_count
    ));

    for (std::size_t i = 0; i < triangle_count; ++i) {
        const Triangle& tri = scene.triangles[i];
        const std::size_t base = i * 3;

        vertices[base * 3 + 0] = tri.v0.x;
        vertices[base * 3 + 1] = tri.v0.y;
        vertices[base * 3 + 2] = tri.v0.z;

        vertices[base * 3 + 3] = tri.v1.x;
        vertices[base * 3 + 4] = tri.v1.y;
        vertices[base * 3 + 5] = tri.v1.z;

        vertices[base * 3 + 6] = tri.v2.x;
        vertices[base * 3 + 7] = tri.v2.y;
        vertices[base * 3 + 8] = tri.v2.z;

        indices[i * 3 + 0] = static_cast<unsigned>(base + 0);
        indices[i * 3 + 1] = static_cast<unsigned>(base + 1);
        indices[i * 3 + 2] = static_cast<unsigned>(base + 2);
    }

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(static_cast<RTCScene>(rtc_scene_), geometry);
    rtcReleaseGeometry(geometry);
    rtcCommitScene(static_cast<RTCScene>(rtc_scene_));

    return true;
}

bool EmbreeScene::intersect(const Ray& ray, float t_min, float t_max, HitInfo& hit) const {
    if (!rtc_scene_ || !scene_) {
        return false;
    }

    RTCRayHit ray_hit;
    ray_hit.ray.org_x = ray.origin.x;
    ray_hit.ray.org_y = ray.origin.y;
    ray_hit.ray.org_z = ray.origin.z;
    ray_hit.ray.dir_x = ray.direction.x;
    ray_hit.ray.dir_y = ray.direction.y;
    ray_hit.ray.dir_z = ray.direction.z;
    ray_hit.ray.tnear = t_min;
    ray_hit.ray.tfar = t_max;
    ray_hit.ray.mask = 0xFFFFFFFFu;
    ray_hit.ray.flags = 0;
    ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    ray_hit.hit.primID = RTC_INVALID_GEOMETRY_ID;

    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);
    rtcIntersect1(static_cast<RTCScene>(rtc_scene_), &ray_hit, &args);

    if (ray_hit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return false;
    }

    const unsigned prim_id = ray_hit.hit.primID;
    if (static_cast<std::size_t>(prim_id) >= scene_->triangles.size()) {
        return false;
    }

    const Triangle& tri = scene_->triangles[prim_id];
    const float t = ray_hit.ray.tfar;
    Vec3 normal{ray_hit.hit.Ng_x, ray_hit.hit.Ng_y, ray_hit.hit.Ng_z};
    if (length(normal) == 0.0f) {
        normal = cross(tri.v1 - tri.v0, tri.v2 - tri.v0);
    }
    normal = normalize(normal);
    if (dot(normal, ray.direction) > 0.0f) {
        normal = -normal;
    }

    hit.hit = true;
    hit.t = t;
    hit.position = ray.origin + ray.direction * t;
    hit.normal = normal;
    hit.material_id = tri.material_id;
    return true;
}

}
