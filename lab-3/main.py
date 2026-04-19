import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(35)

COUNT_LABEL = "Counts in regions:"
EXPECTED_LABEL = "Expected per region:"
ABS_DEV_LABEL = "Absolute deviation:"
REL_DEV_LABEL = "Relative deviation (%):"


def generate_random_triangle_point(p0, p1, p2):
    u, v = p1 - p0, p2 - p0
    e_u, e_v = rng.uniform(), rng.uniform()
    
    if (e_u + e_v > 1):
        e_u = 1 - e_u
        e_v = 1 - e_v
    
    return p0 + u * e_u + v * e_v

def generate_random_circle_point(n, r, c):
    n = n / np.linalg.norm(n)

    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    while True:
        e_u, e_v = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        if (e_u ** 2 + e_v ** 2 <= 1.0):
            break

    return c + r * (u * e_u + v * e_v)


def generate_random_sphere_point(r, c):
    e_phi, e_theta = rng.uniform(), rng.uniform()
    
    phi = 2 * np.pi * e_phi
    theta = np.acos(2 * e_theta - 1)
    
    return c + np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])


def generate_random_cosine_direction(n):
    n = n / np.linalg.norm(n)

    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)

    e_phi, e_theta = rng.uniform(), rng.uniform()
    phi = 2.0 * np.pi * e_phi
    sin_theta = np.sqrt(e_theta)
    cos_theta = np.sqrt(1.0 - e_theta)

    d = u * (np.cos(phi) * sin_theta) + v * (np.sin(phi) * sin_theta) + n * cos_theta
    return d / np.linalg.norm(d)


def _is_point_in_triangle_2d(p, a, b, c, eps=1e-12):
    v0 = b - a
    v1 = c - a
    v2 = p - a

    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < eps:
        return False

    u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
    v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
    w = 1.0 - u - v

    return (u >= -eps) and (v >= -eps) and (w >= -eps)


def analyze_triangle_generator(vertices, sampled_points):
    p0, p1, p2 = vertices
    centroid = (p0 + p1 + p2) / 3.0
    n_points = len(sampled_points)

    regions = [
        (p0, p1, centroid),
        (p1, p2, centroid),
        (p2, p0, centroid),
    ]
    counts = np.zeros(3, dtype=int)

    for point in sampled_points:
        if _is_point_in_triangle_2d(point, *regions[0]):
            counts[0] += 1
        elif _is_point_in_triangle_2d(point, *regions[1]):
            counts[1] += 1
        elif _is_point_in_triangle_2d(point, *regions[2]):
            counts[2] += 1

    expected = n_points / 3.0
    abs_dev = np.abs(counts - expected)
    rel_dev_percent = 100.0 * abs_dev / expected

    print(COUNT_LABEL, counts)
    print(EXPECTED_LABEL, round(expected, 2))
    print(ABS_DEV_LABEL, np.round(abs_dev, 2))
    print(REL_DEV_LABEL, np.round(rel_dev_percent, 2))


def analyze_circle_generator(normal, center, circle_points):
    normal = normal / np.linalg.norm(normal)
    n_points = len(circle_points)

    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, normal)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    rel_points = circle_points - center
    x_local = rel_points @ u
    y_local = rel_points @ v
    angles = np.mod(np.arctan2(y_local, x_local), 2.0 * np.pi)

    sector_index = np.floor(angles / (2.0 * np.pi / 3.0)).astype(int)
    sector_index = np.clip(sector_index, 0, 2)
    counts = np.bincount(sector_index, minlength=3)

    expected = n_points / 3.0
    abs_dev = np.abs(counts - expected)
    rel_dev_percent = 100.0 * abs_dev / expected

    print(COUNT_LABEL, counts)
    print(EXPECTED_LABEL, round(expected, 2))
    print(ABS_DEV_LABEL, np.round(abs_dev, 2))
    print(REL_DEV_LABEL, np.round(rel_dev_percent, 2))


def analyze_sphere_generator(center, sphere_points):
    n_points = len(sphere_points)
    rel_points = sphere_points - center
    rel_points_unit = rel_points / np.linalg.norm(rel_points, axis=1, keepdims=True)

    cap_centers = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    alpha = np.pi / 6.0
    cos_alpha = np.cos(alpha)

    dots = rel_points_unit @ cap_centers.T
    counts = np.sum(dots >= cos_alpha, axis=0)

    cap_area_fraction = (1.0 - cos_alpha) / 2.0
    expected = n_points * cap_area_fraction
    abs_dev = np.abs(counts - expected)
    rel_dev_percent = 100.0 * abs_dev / expected

    print(COUNT_LABEL, counts)
    print(EXPECTED_LABEL, round(expected, 2))
    print(ABS_DEV_LABEL, np.round(abs_dev, 2))
    print(REL_DEV_LABEL, np.round(rel_dev_percent, 2))


def analyze_cosine_generator(normal, directions):
    normal = normal / np.linalg.norm(normal)
    n_points = len(directions)

    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, normal)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    directions_unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    cos_theta = directions_unit @ normal

    cap_half_angle = np.pi / 12.0
    cos_cap = np.cos(cap_half_angle)

    def build_cap_center(polar_angle, azimuth):
        return (
            normal * np.cos(polar_angle)
            + (np.cos(azimuth) * u + np.sin(azimuth) * v) * np.sin(polar_angle)
        )

    cap_centers = np.array([
        build_cap_center(np.pi / 12.0, 0.0),
        build_cap_center(np.pi / 4.0, 2.0 * np.pi / 3.0),
        build_cap_center(5.0 * np.pi / 12.0, 4.0 * np.pi / 3.0),
    ])

    dot_products = directions_unit @ cap_centers.T
    counts = np.sum(dot_products >= cos_cap, axis=0)

    theta = np.linspace(0.0, np.pi / 2.0, 360)
    phi = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    hemi_dirs = (
        np.sin(theta_grid)[..., None] * np.cos(phi_grid)[..., None] * u
        + np.sin(theta_grid)[..., None] * np.sin(phi_grid)[..., None] * v
        + np.cos(theta_grid)[..., None] * normal
    )

    weight = (np.cos(theta_grid) / np.pi) * np.sin(theta_grid)
    total_weight = np.sum(weight)

    cap_probs = []
    for cap_center in cap_centers:
        in_cap = (hemi_dirs @ cap_center) >= cos_cap
        cap_probs.append(np.sum(weight * in_cap) / total_weight)
    cap_probs = np.array(cap_probs)

    expected = n_points * cap_probs
    abs_dev = np.abs(counts - expected)
    rel_dev_percent = 100.0 * abs_dev / expected

    empirical_mean_cos = np.mean(cos_theta)
    theoretical_mean_cos = 2.0 / 3.0
    mean_cos_abs_dev = abs(empirical_mean_cos - theoretical_mean_cos)
    mean_cos_rel_dev_percent = 100.0 * mean_cos_abs_dev / theoretical_mean_cos

    print(COUNT_LABEL, counts)
    print(EXPECTED_LABEL, np.round(expected, 2))
    print(ABS_DEV_LABEL, np.round(abs_dev, 2))
    print(REL_DEV_LABEL, np.round(rel_dev_percent, 2))
    print("Empirical mean cos(theta):", round(empirical_mean_cos, 2))
    print("Theoretical mean cos(theta):", round(theoretical_mean_cos, 2))
    print(ABS_DEV_LABEL, round(mean_cos_abs_dev, 2))
    print(REL_DEV_LABEL, round(mean_cos_rel_dev_percent, 2))

    return {
        "counts": counts,
        "expected_count_per_cap": np.round(expected, 2),
        "absolute_deviation": np.round(abs_dev, 2),
        "relative_deviation_percent": np.round(rel_dev_percent, 2),
        "empirical_mean_cos_theta": round(empirical_mean_cos, 2),
        "theoretical_mean_cos_theta": round(theoretical_mean_cos, 2),
        "mean_cos_theta_absolute_deviation": round(mean_cos_abs_dev, 2),
        "mean_cos_theta_relative_deviation_percent": round(mean_cos_rel_dev_percent, 2),
    }


def plot_triangle_points(vertices, n_points):
    p0, p1, p2 = vertices
    triangle_points = np.array([
        generate_random_triangle_point(p0, p1, p2)
        for _ in range(n_points)
    ])

    analyze_triangle_generator(vertices, triangle_points)

    triangle_vertices = np.vstack([vertices, vertices[0]])

    plt.figure(figsize=(7, 6))
    plt.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], color="black", linewidth=1.5)
    plt.scatter(triangle_points[:, 0], triangle_points[:, 1], s=8, alpha=0.65)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.28, linewidth=0.7)
    plt.title(f"Random points inside a triangle (n={n_points})")
    plt.tight_layout()
    plt.show()


def plot_circle_points(normal, radius, center, n_points):
    circle_points = np.array([
        generate_random_circle_point(normal, radius, center)
        for _ in range(n_points)
    ])

    analyze_circle_generator(normal, center, circle_points)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], s=8, alpha=0.65)
    
    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, normal)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    r = np.linalg.norm(radius)

    t = np.linspace(0.0, 2.0 * np.pi, 200)
    contour = center + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], color="black", linewidth=1.5)
    
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.set_title(f"Random points inside a circle (n={n_points})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


def plot_sphere_points(radius, center, n_points):
    sphere_points = np.array([
        generate_random_sphere_point(radius, center)
        for _ in range(n_points)
    ])

    analyze_sphere_generator(center, sphere_points)

    sphere_points_plot = center + (sphere_points - center) * 1.03

    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 30)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot_surface(x, y, z, color="lightgray", alpha=1.0, linewidth=0)
    
    ax.scatter(sphere_points_plot[:, 0], sphere_points_plot[:, 1], sphere_points_plot[:, 2], s=8, color="blue")
    
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.set_title(f"Random points on a sphere (n={n_points})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


def plot_cosine_directions(normal, center, n_points):
    normal = normal / np.linalg.norm(normal)
    center = np.asarray(center, dtype=float)

    directions = np.array([
        generate_random_cosine_direction(normal)
        for _ in range(n_points)
    ])
    endpoints = center + directions

    analyze_cosine_generator(normal, directions)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], s=8, color="blue", alpha=0.75)
    ax.scatter(center[0], center[1], center[2], s=22, color="black")
    ax.quiver(
        center[0], center[1], center[2],
        normal[0], normal[1], normal[2],
        length=1.2, normalize=True, color="black", linewidth=1.6
    )

    lim = 1.25
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_zlim(center[2] - lim, center[2] + lim)
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.set_title(f"Cosine-weighted directions (n={n_points})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n = 10000

    triangle_vertices = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [3.0, 3.0],
    ])

    circle_normal = np.array([0.0, 0.0, 1.0])
    circle_radius = 2.0
    circle_center = np.array([0.0, 0.0, 0.0])

    sphere_radius = 2.0
    sphere_center = np.array([0.0, 0.0, 0.0])

    cosine_normal = np.array([0.0, 0.0, 1.0])
    cosine_center = np.array([0.0, 0.0, 0.0])

    plot_triangle_points(triangle_vertices, n)
    plot_circle_points(circle_normal, circle_radius, circle_center, n)
    plot_sphere_points(sphere_radius, sphere_center, n)
    plot_cosine_directions(cosine_normal, cosine_center, n)
