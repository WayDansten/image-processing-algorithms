#pragma once

#include <string>

#include "core/scene.h"

namespace pt {

bool load_obj(const std::string& path, Scene& scene, std::string& error_message);

}
