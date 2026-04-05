import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng(35)


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


def plot_triangle_points(vertices, n_points):
    p0, p1, p2 = vertices
    triangle_points = np.array([
        generate_random_triangle_point(p0, p1, p2)
        for _ in range(n_points)
    ])

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
        generate_random_sphere_point(radius, center) * 1.03
        for _ in range(n_points)
    ])

    u = np.linspace(0.0, 2.0 * np.pi, 30)
    v = np.linspace(0.0, np.pi, 30)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot_surface(x, y, z, color="lightgray", alpha=1.0, linewidth=0)
    
    ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], s=8, color="blue")
    
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
    n = 1000

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
