# import numpy as np
# import open3d as o3d

# # pcd = o3d.io.read_point_cloud("/root/results/DynoSAM/incremental_kitti_00_test/object_map_k11_j1.pcd")
# pcd = o3d.io.read_point_cloud("/root/results/DynoSAM/incremental_omd_test/object_map_k100_j2.pcd")

# # pcd = o3d.io.read_point_cloud("/root/results/Dynosam_ecmr2024/cluster_l1/object_map_k82_j24.pcd")
# pcd.paint_uniform_color([0.1, 0.1, 0.7])

# xyz_load = np.asarray(pcd.points)
# print(np.asarray(pcd.colors))

# o3d.visualization.draw_geometries([pcd])

import open3d as o3d
import os
import re
import numpy as np
import time
def load_sorted_object_pcds(directory, object_id='j1', prefix=None):
    regex_pattern = r'object_map_k(\d+)_{}\.pcd'.format(object_id)
    if prefix is not None:
        regex_pattern = prefix + regex_pattern
    pattern = re.compile(regex_pattern)
    pcd_files = []

    for file in os.listdir(directory):
        match = pattern.match(file)
        if match:
            frame_num = int(match.group(1))
            pcd_files.append((frame_num, os.path.join(directory, file)))

    pcd_files.sort(key=lambda x: x[0])
    return pcd_files

def rotate_point_cloud(pcd, angle_deg, axis='y'):
    axis_vector = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis]
    R = pcd.get_rotation_matrix_from_axis_angle(
        angle_deg * np.pi / 180 * np.array(axis_vector)
    )
    center = pcd.get_center()  # <- this
    pcd.rotate(R, center=center)
    return pcd

point_size = 8

def capture_initial_view(pcd):
    print("Adjust the view as desired. Then press 'q' or close the window to start the animation.")
    vis = o3d.visualization.Visualizer()
    vis.create_window("Set View Angle", width=1280, height=720)
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([255, 255, 255])
    render_opt.point_size = point_size

    # Compute center of the point cloud
    center = pcd.get_center()
    # Create coordinate frame at the center
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=center)

    vis.add_geometry(pcd)
    vis.add_geometry(axis_frame)

    vis.run()  # This allows you to manually adjust the view
    view_ctl = vis.get_view_control()
    params = view_ctl.convert_to_pinhole_camera_parameters()
    vis.capture_screen_image('cameraparams.png')
    vis.destroy_window()
    return params

def visualize_spinning_growth(pcd_files, spin_steps=10, total_spin_deg=15, delay=0.01):
    # Load the first point cloud and let user manually adjust view
    initial_pcd = o3d.io.read_point_cloud(pcd_files[-1][1])
    camera_params = capture_initial_view(initial_pcd)

    print(camera_params)

    vis = o3d.visualization.Visualizer()
    window = vis.create_window(window_name='Spinning Object Growth', width=1280, height=720)
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([255, 255, 255])
    render_opt.point_size = point_size
    render_opt.light_on = True  # ✅ Add lighting
    # render_opt.show_coordinate_frame = True


    current_pcd = None
    cumulative_transform = np.eye(4)  # Identity matrix for accumulating transformations

    # Apply captured camera view
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

    for i, (_, pcd_file) in enumerate(pcd_files):
        print(f"Loading frame {i + 1}/{len(pcd_files)}: {pcd_file}")
        new_pcd = o3d.io.read_point_cloud(pcd_file)
        new_pcd, _ = new_pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=2.0)

         # ✅ Estimate normals for lighting
        new_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        new_pcd.transform(cumulative_transform)

        # Apply cumulative transform so it starts where the last left off

        if current_pcd is not None:
            vis.remove_geometry(current_pcd)

        current_pcd = new_pcd
        vis.add_geometry(current_pcd)


        for step in range(spin_steps):
            # Compute incremental rotation
            # angle_deg = total_spin_deg / spin_steps
            # axis_vector = np.array([0, 1, 0])  # spinning-top: z-axis
            # rotation_matrix = current_pcd.get_rotation_matrix_from_axis_angle(
            #     angle_deg * np.pi / 180 * axis_vector
            # )
            # center = current_pcd.get_center()

            # # Compose transformation: translate to origin -> rotate -> back to center
            # T = np.eye(4)
            # T[:3, :3] = rotation_matrix
            # translate_to_origin = np.eye(4)
            # translate_to_origin[:3, 3] = -center
            # translate_back = np.eye(4)
            # translate_back[:3, 3] = center
            # full_transform = translate_back @ T @ translate_to_origin

            # current_pcd.transform(full_transform)
            # cumulative_transform = full_transform @ cumulative_transform  # update accumulated transform

            vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

            vis.update_geometry(current_pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(delay)

    print("Final frame reached. Press Q or ESC to exit.")
    vis.run()
    vis.destroy_window()


# Example usage:
# if __name__ == "__main__":
#     show_ply_with_indirect_lighting("/root/results/DynoSAM/incremental_omd_test/object_map_k119_j2.pcd")

if __name__ == "__main__":
    # directory = "/root/results/DynoSAM/incremental_omd_test/"
    # directory = "/root/results/Dynosam_ecmr2024/cluster_l1_map/"
    # directory = "/root/results/Dynosam_ecmr2024/test_kitti_04/"
    directory = "/root/results/frontend_filtering/omd_MO_test1/"
    prefix = "doo_"
    # prefix = "refined_"
    object_id = "j2"
    pcd_files = load_sorted_object_pcds(directory, object_id, prefix=prefix)
    visualize_spinning_growth(pcd_files)
