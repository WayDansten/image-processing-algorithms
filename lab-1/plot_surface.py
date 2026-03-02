import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def visualize_triangle_illumination(mesh_data_file='mesh_data.npz', output_file='visualization.png'):
    data = np.load(mesh_data_file)
    vertices = data['vertices']
    brightness = data['brightness']
    u_grid = data['u_grid']
    v_grid = data['v_grid']
    lights_color_1 = data['lights_color_1']
    lights_position_1 = data['lights_position_1']
    lights_color_2 = data['lights_color_2']
    lights_position_2 = data['lights_position_2']
    
    triangle_mask = (u_grid + v_grid) <= 1.0
    
    x = vertices[:, :, 0].copy()
    y = vertices[:, :, 1].copy()
    z = vertices[:, :, 2].copy()
    rgb = brightness.copy()
    
    x[~triangle_mask] = np.nan
    y[~triangle_mask] = np.nan
    z[~triangle_mask] = np.nan
    rgb[~triangle_mask] = np.nan
    
    dpi = 100
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(x, y, z, facecolors=rgb, shade=False)
    
    light1_color_norm = lights_color_1 / 255.0
    light2_color_norm = lights_color_2 / 255.0
    
    ax.scatter([lights_position_1[0]], [lights_position_1[1]], [lights_position_1[2]], 
               color=light1_color_norm, s=100, marker='o', edgecolors='black', linewidth=1.5)
    ax.scatter([lights_position_2[0]], [lights_position_2[1]], [lights_position_2[2]], 
               color=light2_color_norm, s=100, marker='o', edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        mesh_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) >= 3 else 'visualization.png'
    else:
        mesh_file = input("Enter mesh data file (default: mesh_data.npz): ").strip()
        if not mesh_file:
            mesh_file = 'mesh_data.npz'
        
        output_file = input("Enter output visualization file (default: visualization.png): ").strip()
        if not output_file:
            output_file = 'visualization.png'
    
    visualize_triangle_illumination(mesh_data_file=mesh_file, output_file=output_file)
