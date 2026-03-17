import numpy as np
from tabulate import tabulate
import subprocess
import sys
import os
import math

def normalize(v):
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm <= 1e-10:
        return np.zeros_like(v)
    return v / norm

def calculate_light_power_at_angle(I0, cos_theta):
    I0 = np.array(I0, dtype=float)
    d_theta = np.clip(np.abs(cos_theta), 0.0, 1.0)
    return I0 * d_theta

def calculate_illumination(I_theta, cos_alpha, R):
    I_theta = np.array(I_theta, dtype=float)
    cos_alpha = np.clip(np.abs(cos_alpha), 0.0, 1.0)
    E = I_theta * cos_alpha / (R * R + 1e-10)
    return E

def calculate_triangle_normal(triangle):
    v0 = np.array(triangle[0], dtype=float)
    v1 = np.array(triangle[1], dtype=float)
    v2 = np.array(triangle[2], dtype=float)
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    return normalize(normal)

def local_to_global(triangle, x, y):
    v0 = np.array(triangle[0], dtype=float)
    v1 = np.array(triangle[1], dtype=float)
    v2 = np.array(triangle[2], dtype=float)
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    point = v0 + x * edge1 + y * edge2
    return point

def build_local_frame(triangle):
    v0 = np.array(triangle[0], dtype=float)
    v1 = np.array(triangle[1], dtype=float)
    v2 = np.array(triangle[2], dtype=float)

    edge1 = v1 - v0
    edge2 = v2 - v0

    x_axis = normalize(edge1)
    z_axis = normalize(np.cross(edge1, edge2))
    y_axis = normalize(np.cross(z_axis, x_axis))

    rotation_local_to_global = np.column_stack((x_axis, y_axis, z_axis))
    return v0, rotation_local_to_global

def global_point_to_local(point, origin, rotation_local_to_global):
    point = np.array(point, dtype=float)
    return rotation_local_to_global.T @ (point - origin)

def global_vector_to_local(vector, rotation_local_to_global):
    vector = np.array(vector, dtype=float)
    return rotation_local_to_global.T @ vector

def calculate_half_vector(light_dir, view_dir):
    light_dir = np.array(light_dir, dtype=float)
    view_dir = np.array(view_dir, dtype=float)
    h = light_dir + view_dir
    return normalize(h)

def calculate_brdf(light_dir, view_dir, normal, k_d, k_s, n, surface_color):
    light_dir = normalize(light_dir)
    view_dir = normalize(view_dir)
    normal = normalize(normal)
    surface_color = np.array(surface_color, dtype=float)
    
    h = calculate_half_vector(light_dir, view_dir)
    
    nh = np.dot(normal, h)
    
    diffuse = k_d
    specular = k_s * (nh ** n)
    
    brdf = surface_color * (diffuse + specular)
    return brdf

def calculate_brightness_at_point(
    point,
    normal,
    lights,
    view_dir,
    k_d,
    k_s,
    n,
    surface_color,
):
    point = np.array(point, dtype=float)
    normal = normalize(normal)
    view_dir = normalize(view_dir)
    
    E_total = np.zeros(3, dtype=float)
    B_total = np.zeros(3, dtype=float)
    
    for light in lights:
        light_pos = np.array(light['position'], dtype=float)
        light_axis = normalize(light['axis'])
        light_color = np.array(light['color'], dtype=float)
        
        source_to_point = point - light_pos
        R = np.linalg.norm(source_to_point)
        if R <= 1e-10:
            continue
        
        source_to_point_dir = normalize(source_to_point)
        point_to_light_dir = -source_to_point_dir
        
        cos_theta = np.dot(source_to_point_dir, light_axis)
        I_theta = calculate_light_power_at_angle(light_color, cos_theta)
        
        cos_alpha = np.dot(point_to_light_dir, normal)
        E = calculate_illumination(I_theta, cos_alpha, R)
        
        brdf = calculate_brdf(source_to_point_dir, view_dir, normal, k_d, k_s, n, surface_color)
        B = E * brdf
        
        E_total += E
        B_total += B
    
    E = np.clip(E_total, 0, None)
    B = np.clip(B_total, 0, None) / math.pi
    
    return E, B

def generate_grid_points(u_values, v_values):
    points = []
    for v in v_values:
        row = []
        for u in u_values:
            row.append((u, v))
        points.append(row)
    return np.array(points, dtype=object)

def generate_dense_triangle_mesh(triangle, normal, lights, view_dir, k_d, k_s, n, surface_color, resolution=50):
    u_range = np.linspace(0.0, 1.0, resolution)
    v_range = np.linspace(0.0, 1.0, resolution)
    
    u_grid, v_grid = np.meshgrid(u_range, v_range)
    
    vertices = np.zeros((resolution, resolution, 3), dtype=float)
    brightness = np.zeros((resolution, resolution, 3), dtype=float)
    
    for i in range(resolution):
        for j in range(resolution):
            u = u_grid[i, j]
            v = v_grid[i, j]
            
            point = local_to_global(triangle, u, v)
            B = calculate_brightness_at_point(
                point=point,
                normal=normal,
                lights=lights,
                view_dir=view_dir,
                k_d=k_d,
                k_s=k_s,
                n=n,
                surface_color=surface_color,
            )[1]
            
            vertices[i, j] = point
            brightness[i, j] = np.clip(B, 0, 1.0)
    
    return {
        'vertices': vertices,
        'brightness': brightness,
        'u_grid': u_grid,
        'v_grid': v_grid
    }

def format_rgb(rgb):
    rgb = np.array(rgb, dtype=float)
    rgb_255 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(int)
    return f"({rgb_255[0]}, {rgb_255[1]}, {rgb_255[2]})"

def save_tables_to_file(filename, E_local_table, E_global_table, B_table, headers):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== Table 1: Illumination E (Local Coordinates) ===\n")
        f.write(tabulate(E_local_table, headers=headers, tablefmt="fancy_grid"))
        f.write("\n\n")
        
        f.write("=== Table 2: Illumination E (Global Coordinates) ===\n")
        f.write(tabulate(E_global_table, headers=headers, tablefmt="fancy_grid"))
        f.write("\n\n")
        
        f.write("=== Table 3: Brightness B ===\n")
        f.write(tabulate(B_table, headers=headers, tablefmt="fancy_grid"))
        f.write("\n")

def parse_input(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    data = {}
    idx = 0
    
    light1_color = np.array(list(map(float, lines[idx].split())))
    idx += 1
    light2_color = np.array(list(map(float, lines[idx].split())))
    idx += 1
    
    light1_axis = np.array(list(map(float, lines[idx].split())))
    idx += 1
    light2_axis = np.array(list(map(float, lines[idx].split())))
    idx += 1
    
    light1_pos = np.array(list(map(float, lines[idx].split())))
    idx += 1
    light2_pos = np.array(list(map(float, lines[idx].split())))
    idx += 1
    
    data['triangle'] = np.array([
        list(map(float, lines[idx].split())),
        list(map(float, lines[idx+1].split())),
        list(map(float, lines[idx+2].split()))
    ])
    idx += 3
    
    data['obs_params'] = np.array([
        list(map(float, lines[idx].split())),
        list(map(float, lines[idx+1].split()))
    ])
    idx += 2
    
    data['obs_direction'] = np.array(list(map(float, lines[idx].split())))
    idx += 1
    
    data['surface_color'] = np.array(list(map(float, lines[idx].split())))
    idx += 1
    
    data['brdf_diffuse'] = float(lines[idx])
    idx += 1
    
    data['brdf_specular_coeff'] = float(lines[idx])
    idx += 1
    
    data['specular_exponent'] = float(lines[idx])
    
    data['lights'] = [
        {
            'color': light1_color,
            'position': light1_pos,
            'axis': light1_axis
        },
        {
            'color': light2_color,
            'position': light2_pos,
            'axis': light2_axis
        }
    ]
    
    return data

if __name__ == "__main__":
    
    files = [
        "input_general.txt",
        "input_colored.txt",
        "input_large_distance.txt",
        "input_black_surface.txt",
        "input_specular_reflection.txt",
    ]
    
    print("Available input files:")
    for i in range(len(files)):
        print(f"  [{i+1}] {files[i]}")
    
    
    filename = files[int(input(f"Select input file (1-{len(files)}): ").strip()) - 1]
    if not filename:
        filename = 'input_general.txt'
    
    input_path = f'input_samples/{filename}'
    data = parse_input(input_path)
    triangle = data['triangle']
    normal = calculate_triangle_normal(triangle)

    view_dir = normalize(data['obs_direction'])
    k_d = data['brdf_diffuse']
    k_s = data['brdf_specular_coeff']
    n = data['specular_exponent']
    surface_color = data['surface_color']

    u_values = data['obs_params'][0]
    v_values = data['obs_params'][1]

    local_origin, rotation_local_to_global = build_local_frame(triangle)
    normal_local = global_vector_to_local(normal, rotation_local_to_global)
    view_dir_local = global_vector_to_local(view_dir, rotation_local_to_global)
    lights_local = []
    for light in data['lights']:
        lights_local.append({
            'color': np.array(light['color'], dtype=float),
            'position': global_point_to_local(light['position'], local_origin, rotation_local_to_global),
            'axis': global_vector_to_local(light['axis'], rotation_local_to_global)
        })
    
    grid_points = generate_grid_points(u_values, v_values)
    
    E_local_table = []
    E_global_table = []
    B_table = []
    E_local_values = np.zeros((len(v_values), len(u_values), 3), dtype=float)
    E_global_values = np.zeros((len(v_values), len(u_values), 3), dtype=float)
    
    for i, row in enumerate(grid_points):
        E_local_row = [f"v={v_values[i]:.2f}"]
        E_global_row = [f"v={v_values[i]:.2f}"]
        B_row = [f"v={v_values[i]:.2f}"]
        
        for j, (u, v) in enumerate(row):
            point_global = local_to_global(triangle, u, v)
            point_local = global_point_to_local(point_global, local_origin, rotation_local_to_global)

            E_local, _ = calculate_brightness_at_point(
                point=point_local,
                normal=normal_local,
                lights=lights_local,
                view_dir=view_dir_local,
                k_d=k_d,
                k_s=k_s,
                n=n,
                surface_color=surface_color,
            )

            E_global, B = calculate_brightness_at_point(
                point=point_global,
                normal=normal,
                lights=data['lights'],
                view_dir=view_dir,
                k_d=k_d,
                k_s=k_s,
                n=n,
                surface_color=surface_color,
            )

            E_local_values[i, j] = E_local
            E_global_values[i, j] = E_global
            
            E_local_row.append(format_rgb(E_local))
            E_global_row.append(format_rgb(E_global))
            B_row.append(format_rgb(B))
        
        E_local_table.append(E_local_row)
        E_global_table.append(E_global_row)
        B_table.append(B_row)
    
    headers = [""] + [f"u={u:.2f}" for u in u_values]
    
    print("\n=== Table 1: Illumination E (Local Coordinates) ===")
    print(tabulate(E_local_table, headers=headers, tablefmt="fancy_grid"))
    
    print("\n=== Table 2: Illumination E (Global Coordinates) ===")
    print(tabulate(E_global_table, headers=headers, tablefmt="fancy_grid"))
    
    print("\n=== Table 3: Brightness B ===")
    print(tabulate(B_table, headers=headers, tablefmt="fancy_grid"))
    
    if not os.path.exists("output"):
        os.makedirs("output")

    save_tables_to_file("output/output.txt", E_local_table, E_global_table, B_table, headers)
    
    print("\nGenerating dense triangle mesh for visualization...")
    mesh_data = generate_dense_triangle_mesh(
        triangle=triangle,
        normal=normal,
        lights=data['lights'],
        view_dir=view_dir,
        k_d=k_d,
        k_s=k_s,
        n=n,
        surface_color=surface_color,
        resolution=50
    )
    print(f"Generated {mesh_data['vertices'].shape[0]}x{mesh_data['vertices'].shape[1]} mesh grid")

    np.savez('output/mesh_data.npz',
             vertices=mesh_data['vertices'],
             brightness=mesh_data['brightness'],
             u_grid=mesh_data['u_grid'],
             v_grid=mesh_data['v_grid'],
             lights_color_1=data['lights'][0]['color'],
             lights_position_1=data['lights'][0]['position'],
             lights_color_2=data['lights'][1]['color'],
             lights_position_2=data['lights'][1]['position'],
             triangle=triangle)
    print("Mesh data saved to output/mesh_data.npz")
    
    print("\nGenerating 3D visualization...")
    subprocess.run([sys.executable, 'plot_surface.py', 'output/mesh_data.npz', 'output/visualization.png'])
    print("\nAll outputs generated successfully!")
    print("  - output/output.txt (tables)")
    print("  - output/mesh_data.npz (mesh data)")
    print("  - output/visualization.png (3D visualization)")
