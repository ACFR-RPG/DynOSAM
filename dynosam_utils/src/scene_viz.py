import open3d as o3d
import numpy as np
import os

PATH_TO_MODELS = "/root/data/models/"

def create_stylized_car(color, model_name="sk_audi_a2.stl"):
    """Loads an external car mesh, centers it, and paints it uniformly."""
    mesh_path = os.path.join(PATH_TO_MODELS, model_name)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh not found at: {mesh_path}")

    car_mesh = o3d.io.read_triangle_mesh(mesh_path)

    # 1. Paint the mesh with the timestep color
    car_mesh.paint_uniform_color(color)

    # 2. Recompute normals so the flat/matte shading looks clean
    car_mesh.compute_vertex_normals()


    # Optional: If the STL model arrives huge or tiny, scale it to match the scene grid unit.
    # Un-comment the line below if you need to force its maximum dimension to be ~2.0 units.
    car_mesh.scale(2.0 / np.max(car_mesh.get_max_bound() - car_mesh.get_min_bound()), center=(0, 0, 0))

     # 3. Center the car base geometry at (0,0,0) so the trajectory rotations (Yaw)
    # happen predictably around the car's own center.
    offset = -car_mesh.get_center()
    offset[2] += 0.4
    car_mesh.translate(offset)

    # 2. FIX ROLL OFFSET: Rotate -90 degrees around the X-axis
    # (If the car turns upside down instead, change -90 to 90)
    R_fix = car_mesh.get_rotation_matrix_from_xyz((np.radians(-90), 0, 0))
    car_mesh.rotate(R_fix, center=(0, 0, 0))

    return car_mesh

def create_grid_floor(size=10, n=10):
    """Creates a clean coordinate grid for the static background."""
    lines = []
    points = []

    # Generate grid lines
    vals = np.linspace(-size/2, size/2, n)
    idx = 0
    for v in vals:
        # Lines parallel to X-axis
        points.append([-size/2, 0, v])
        points.append([size/2, 0, v])
        lines.append([idx, idx+1])
        idx += 2

        # Lines parallel to Z-axis
        points.append([v, 0, -size/2])
        points.append([v, 0, size/2])
        lines.append([idx, idx+1])
        idx += 2

    colors = [[0.7, 0.7, 0.7] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

# def generate_snaking_trajectory():
#     """
#     Generates 3 poses along a parameterized sine wave curve.
#     Uses the curve derivative to compute the proper Yaw alignment.
#     """
#     # 3 discrete evaluation points along the curve (from left to right)
#     # t_steps = np.array([-4.0, 0.0, 4.0])
#     t_steps = np.array([-2.0, 0.0, 2.0])

#     # Define sine wave parameters: Amplitude and Frequency
#     amp = 2.0
#     freq = 0.2

#     poses = []
#     for x in t_steps:
#         # Ground plane is X and Z. Y is height (kept flat at 0).
#         z = amp * np.sin(freq * x)

#         # Derivative dz/dx to find the tangent vector for heading/yaw
#         dz_dx = amp * freq * np.cos(freq * x)
#         yaw_rad = np.arctan2(dz_dx, 1.0)
#         yaw_deg = np.degrees(yaw_rad)

#         # Open3D coordinate mapping: X=Forward/Side, Y=Up(0), Z=Depth/Horizontal
#         # We negate the yaw angle depending on your specific STL model's initial forward axis direction.
#         poses.append([x, 0.0, z, -yaw_deg])

#     return poses

def generate_curved_trajectory():
    """
    Generates 3 poses along a single, continuous parabola curve (z = a*x^2 + c).
    Uses the curve derivative to compute the proper Yaw alignment.
    """
    # 3 discrete timesteps positioned along the X-axis
    t_steps = np.array([-2.2, 0.0, 2.5])

    # Parabola configuration: z = a * x^2 + c
    a = 0.15   # Controls the curvature sharpness
    c = -2.0   # Constant offset along the Z axis to center the curve visually

    poses = []
    for x in t_steps:
        # Calculate coordinate on the X-Z ground plane (Y is flat floor)
        z = a * (x**2) + c

        # Take derivative (dz/dx = 2 * a * x) to get the path tangent vector
        dz_dx = 2 * a * x
        yaw_rad = np.arctan2(dz_dx, 1.0)
        yaw_deg = np.degrees(yaw_rad)

        # Poses format: [X, Y, Z, Yaw]
        poses.append([x, 0.0, z, -yaw_deg])

    return poses

def main():
    # 1. Initialize Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Dynamic SLAM Poster Asset", width=1200, height=1200)
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1, 1, 1])
    render_opt.line_width = 2.0

    # 2. Add static background
    grid = create_grid_floor(size=14, n=15)
    vis.add_geometry(grid)

    # 3. Define timestep colors (Cool -> Warm trajectory)
    # colors = [
    #     [0.2, 0.6, 0.9],  # t1: Cyan/Blue
    #     [0.5, 0.2, 0.8],  # t2: Purple
    #     [0.9, 0.1, 0.5]   # t3: Magenta
    # ]
    colors = [
        [0.2, 0.6, 0.9],  # t1: Cyan/Blue
        [0.2, 0.6, 0.9],  # t2: Purple
        [0.2, 0.6, 0.9]   # t3: Magenta
    ]

    # 4. Generate dynamic trajectory poses dynamically [X, Y, Z, Yaw]
    poses = generate_curved_trajectory()

    # 5. Generate and place the cars
    for i, (color, pose) in enumerate(zip(colors, poses)):
        car = create_stylized_car(color)

        # Apply orientation (Rotation around Y-axis for Yaw)
        R = car.get_rotation_matrix_from_xyz((0, np.radians(pose[3]), 0))
        car.rotate(R, center=(0, 0, 0))

        # Translate to trajectory position
        car.translate(np.array([pose[0], pose[1], pose[2]]))

        vis.add_geometry(car)

    # 6. Set up an optimized poster-ready isometric viewpoint
    ctr = vis.get_view_control()
    ctr.set_front([-0.5, 0.6, 0.6])  # High angle perspective looking down
    ctr.set_lookat([0.0, 0.0, 0.0])   # Center of the world
    ctr.set_up([0.0, 1.0, 0.0])       # Y-axis is Up
    ctr.set_zoom(0.75)

    print("Mesh rendered along snaking trajectory. Press 'Q' to exit.")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()

# def main():
#     # 1. Initialize Visualizer with a clean white background
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name="Dynamic SLAM Poster Asset", width=1200, height=1200)
#     render_opt = FrameworkRenderOption = vis.get_render_option()
#     render_opt.background_color = np.array([1, 1, 1]) # Pure white for poster
#     render_opt.line_width = 2.0

#     # 2. Add static background (Grid Floor)
#     grid = create_grid_floor(size=12, n=13)
#     vis.add_geometry(grid)

#     # 3. Define trajectories/poses for the object (Car) at 3 timesteps
#     # Timestep colors: Progressing from cool to warm (e.g., Cyan -> Purple -> Magenta)
#     colors = [
#         [0.2, 0.6, 0.9],  # t1: Cyan/Blue
#         [0.5, 0.2, 0.8],  # t2: Purple
#         [0.9, 0.1, 0.5]   # t3: Magenta
#     ]

#     # Poses: [X, Y, Z, Yaw (degrees)]
#     # Moving diagonally across the frame
#     poses = [
#         [-3.0, 0.0, -2.0, 15],   # t1
#         [ 0.0, 0.0, -0.5, 30],   # t2
#         [ 3.0, 0.0,  1.0, 45]    # t3
#     ]

#     # 4. Generate and place the cars
#     for i, (color, pose) in enumerate(zip(colors, poses)):
#         car = create_stylized_car(color)

#         # Apply transformation (Rotation then Translation)
#         R = car.get_rotation_matrix_from_xyz((0, np.radians(pose[3]), 0))
#         car.rotate(R, center=(0, 0, 0))
#         car.translate(np.array([pose[0], pose[1], pose[2]]))

#         vis.add_geometry(car)

#     # 5. Set up a good isometric/orthographic-style viewport looking down at the scene
#     ctr = vis.get_view_control()

#     # Position the camera to look down at the center from a front-left angle
#     ctr.set_front([-0.6, 0.6, 0.5])  # Look direction
#     ctr.set_lookat([0.0, 0.5, 0.0])   # Focus point
#     ctr.set_up([0.0, 1.0, 0.0])       # Up vector
#     ctr.set_zoom(0.7)

#     print("Mesh rendered. Press 'Q' or close the window to exit.")
#     vis.run()
#     vis.destroy_window()

# if __name__ == "__main__":
#     main()
