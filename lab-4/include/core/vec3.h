#pragma once

#include <algorithm>
#include <cmath>

namespace pt {

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float x_value, float y_value, float z_value) : x(x_value), y(y_value), z(z_value) {}

    Vec3 operator+(const Vec3& other) const { return Vec3{x + other.x, y + other.y, z + other.z}; }
    Vec3 operator-(const Vec3& other) const { return Vec3{x - other.x, y - other.y, z - other.z}; }
    Vec3 operator*(float scalar) const { return Vec3{x * scalar, y * scalar, z * scalar}; }
    Vec3 operator/(float scalar) const { return Vec3{x / scalar, y / scalar, z / scalar}; }
    Vec3 operator-() const { return Vec3{-x, -y, -z}; }

    Vec3 operator*(const Vec3& other) const { return Vec3{x * other.x, y * other.y, z * other.z}; }

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

inline Vec3 operator*(float scalar, const Vec3& v) { return v * scalar; }

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline float length(const Vec3& v) { return std::sqrt(dot(v, v)); }

inline Vec3 normalize(const Vec3& v) {
    const float len = length(v);
    if (len == 0.0f) {
        return Vec3{};
    }
    return v / len;
}

inline Vec3 clamp01(const Vec3& v) {
    return Vec3{
        std::clamp(v.x, 0.0f, 1.0f),
        std::clamp(v.y, 0.0f, 1.0f),
        std::clamp(v.z, 0.0f, 1.0f)
    };
}

inline Vec3 clamp_max(const Vec3& v, float max_value) {
    return Vec3{
        std::min(v.x, max_value),
        std::min(v.y, max_value),
        std::min(v.z, max_value)
    };
}

inline float max_component(const Vec3& v) {
    return std::max(v.x, std::max(v.y, v.z));
}

inline Vec3 reflect(const Vec3& direction, const Vec3& normal) {
    return direction - 2.0f * dot(direction, normal) * normal;
}

}
