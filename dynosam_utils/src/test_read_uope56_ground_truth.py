import numpy as np
import h5py
import os
import glob
import cv2
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
import numpy as np
import h5py
import os
import glob
import cv2
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
ground_truth_folder = "/root/data/dynoepts/UOPE56/UOPE56_groundtruth"
ground_truth_obj_folder = "/root/data/dynoepts/UOPE56/UOPE56_obj_pose"
sequence_folder = "/root/data/dynoepts/UOPE56/others_20-29-002"
sequence_name = "others_25"

ground_truth_file_path = os.path.join(ground_truth_folder, f"{sequence_name}.h5")
sequence_file_path = os.path.join(sequence_folder, sequence_name)

cam_ann_path = os.path.join(sequence_file_path, "cam_annotations")
rgb_path = os.path.join(sequence_file_path, "color")
intrinsics_file = os.path.join(sequence_file_path, "intrinsics.txt")

# -----------------------
# Helper functions
# -----------------------
def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def load_intrinsics_custom(path):
    vals = np.fromfile(path, sep=' ')
    fx, fy, cx, cy, skew = vals[:5]
    H, W = int(vals[6]), int(vals[7])
    K = np.array([
        [fx, skew, cx],
        [0,  fy,   cy],
        [0,   0,    1]
    ])
    return K, (H, W)

def draw_axes(img, K, T_c_o, length=0.05):
    axes = np.array([
        [length,0,0],
        [0,length,0],
        [0,0,length]
    ])
    origin = np.zeros((1,3))
    points = np.vstack([origin, axes])
    pts_cam = (T_c_o[:3,:3] @ points.T + T_c_o[:3,3:4]).T
    proj = (K @ pts_cam.T).T
    proj = proj[:,:2] / proj[:,2:3]
    origin_pt = tuple(proj[0].astype(int))
    colors = [(0,0,255),(0,255,0),(255,0,0)]  # X-red, Y-green, Z-blue
    for i in range(3):
        pt = tuple(proj[i+1].astype(int))
        cv2.line(img, origin_pt, pt, colors[i], 2)
    return img

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0])
    y_range = abs(y_limits[1]-y_limits[0])
    z_range = abs(z_limits[1]-z_limits[0])
    max_range = max([x_range, y_range, z_range]) / 2.0
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)

def load_gt_obj_pose(sequence_name: str, ground_truth_obj_folder: str):
    import pickle
    import bz2
    """
    Load ground-truth object poses in camera frame from a .pkl or .pkl.bz2 file.

    Returns:
        obj_data: dict with keys ['sequence_id', 'frame_id', 'gt_pose_o2c']
    """
    pkl_path = os.path.join(ground_truth_obj_folder, f"{sequence_name}.pkl")
    pkl_bz2_path = os.path.join(ground_truth_obj_folder, f"{sequence_name}.pkl.bz2")

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            obj_data = pickle.load(f)
    elif os.path.exists(pkl_bz2_path):
        with bz2.open(pkl_bz2_path, 'rb') as f:
            obj_data = pickle.load(f)
    else:
        raise FileNotFoundError(f"No .pkl or .pkl.bz2 file found for sequence {sequence_name}")

    # Load pickle
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data_list = pickle.load(f)
    elif os.path.exists(pkl_bz2_path):
        with bz2.open(pkl_bz2_path, 'rb') as f:
            data_list = pickle.load(f)
    else:
        raise FileNotFoundError(f"No .pkl or .pkl.bz2 file found for sequence {sequence_name}")

    # Check format
    if not isinstance(data_list, list) or len(data_list) == 0:
        raise ValueError("Pickle does not contain a non-empty list of dicts")

    # Allocate arrays
    N = len(data_list)
    gt_pose_o2c = np.zeros((N, 4, 4))
    frame_id = np.zeros((N, 1), dtype=int)

    for i, entry in enumerate(data_list):
        gt_pose_o2c[i] = entry['gt_pose_o2c']
        frame_id[i, 0] = entry['frame_id']

    print(f"Loaded {N} object poses from sequence '{sequence_name}'")
    print("gt_pose_o2c shape:", gt_pose_o2c.shape)
    print("frame_id shape:", frame_id.shape)

    return gt_pose_o2c, frame_id


# -----------------------
# Load ground truth camera poses
# -----------------------
with h5py.File(ground_truth_file_path, 'r') as f:
    T_w_g = np.asarray(f["gripper_to_world"][:])
    T_c_g = np.asarray(f["gripper_to_cam_constant"][:])
    # seq_name = f["sequence_name"][()]
T_g_c = invert_transform(T_c_g)
T_w_c = np.array([T_w_g[i] @ T_g_c for i in range(len(T_w_g))])
traj_cam = T_w_c[:, :3, 3]

T_o_c, frame_id = load_gt_obj_pose(sequence_name, ground_truth_obj_folder)
print(T_o_c.shape)

# Compute object pose in world frame
T_w_o = np.array([T_w_c[i] @ T_o_c[i] for i in range(T_o_c.shape[0])])
traj_obj = T_w_o[:, :3, 3]

# # -----------------------
# # 3D Trajectory plot
# # -----------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(traj_cam[:,0], traj_cam[:,1], traj_cam[:,2], 'b-', label='Camera (GT)')
ax.plot(traj_obj[:,0], traj_obj[:,1], traj_obj[:,2], 'r-', label='Object')

ax.scatter(traj_cam[0,0], traj_cam[0,1], traj_cam[0,2], c='b', marker='o', label='Camera Start')
ax.scatter(traj_cam[-1,0], traj_cam[-1,1], traj_cam[-1,2], c='b', marker='x', label='Camera End')
ax.scatter(traj_obj[0,0], traj_obj[0,1], traj_obj[0,2], c='r', marker='o', label='Object Start')
ax.scatter(traj_obj[-1,0], traj_obj[-1,1], traj_obj[-1,2], c='r', marker='x', label='Object End')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Camera and Object Trajectories")
ax.legend()
set_axes_equal(ax)
plt.show()

# # -----------------------
# # Load intrinsics
# -----------------------
K, (H_img, W_img) = load_intrinsics_custom(intrinsics_file)
print("K:\n", K)
print("Image size:", W_img, H_img)

# -----------------------
# Load color images
# -----------------------
img_files = sorted(glob.glob(os.path.join(rgb_path, "*.png")))

# -----------------------
# Per-frame visualization with axes
# -----------------------
for i, img_file in enumerate(img_files):
    img = cv2.imread(img_file)
    if img is None:
        continue
    img = cv2.resize(img, (W_img, H_img))

    T_c_w = T_w_c[i]
    T_w_c_inv = invert_transform(T_c_w)

    T_c_o = T_w_c_inv @ T_w_o[i]

    img_axes = draw_axes(img, K, T_c_o, length=0.05)

    cv2.imshow("Camera + Object Axes", img_axes)
    key = cv2.waitKey(50)
    if key == 27:
        break

cv2.destroyAllWindows()
