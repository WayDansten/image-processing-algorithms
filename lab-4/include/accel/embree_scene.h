#pragma once

#include <string>

#include "core/hit.h"
#include "core/ray.h"
#include "core/scene.h"

namespace pt {

class EmbreeScene {
public:
    EmbreeScene();
    ~EmbreeScene();

    EmbreeScene(const EmbreeScene&) = delete;
    EmbreeScene& operator=(const EmbreeScene&) = delete;

    bool build(const Scene& scene, std::string& error_message);
    bool intersect(const Ray& ray, float t_min, float t_max, HitInfo& hit) const;

private:
    const Scene* scene_ = nullptr;
    void* device_ = nullptr;
    void* rtc_scene_ = nullptr;
};

}
