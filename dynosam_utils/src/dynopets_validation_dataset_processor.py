import argparse
from pathlib import Path

from evo.core.sync import associate_trajectories
from evo.core.trajectory import PoseTrajectory3D

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GT_BBOX_COLOR = (0, 205, 102)
MOCAP_BBOX_COLOR = (0, 0, 255)
AXIS_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


# =========================
# Geometry + IO
# =========================

def invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def quaternion_to_rotation_matrix(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Quaternion norm is zero")
    qx, qy, qz, qw = quaternion / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )



def tum_pose_to_transform(pose_row):
    pose_row = np.asarray(pose_row, dtype=np.float64)

    timestamp = pose_row[0]
    if pose_row.shape[0] != 8:
        raise ValueError(f"Expected 8 pose values, got {pose_row.shape[0]}")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(pose_row[4:8])
    transform[:3, 3] = pose_row[1:4]
    return timestamp, transform


def tum_poses_to_transforms(pose_columns):
    # Allocate arrays
    N = len(pose_columns)
    T = np.zeros((N, 4, 4))
    timestamps = np.zeros((N, 1), dtype=np.float64)

    for i, row in enumerate(pose_columns):
        t, pose = tum_pose_to_transform(row)
        timestamps[i] = t
        T[i] = pose
    return timestamps.flatten(), T


def load_intrinsics(intrinsics_file):
    intrinsics = np.loadtxt(str(intrinsics_file))
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    return fx, fy, cx, cy, camera_matrix, dist_coeffs


def load_model(path):
    if path.suffix == ".obj":
        mesh = o3d.io.read_triangle_mesh(str(path))
        return np.asarray(mesh.vertices)
    else:
        pc = o3d.io.read_point_cloud(str(path))
        return np.asarray(pc.points)


def resolve_model_path(data_dir):
    model_dir = data_dir / "model"
    for ext in ["*.obj", "*.ply"]:
        files = list(model_dir.glob(ext))
        if files:
            return files[0]
    raise FileNotFoundError("No model found")


def get_model_scale(path):
    pts = load_model(path)
    return np.diag(np.max(np.abs(pts), axis=0))


def load_camera_poses(path):
    data = np.load(path)

    poses = data["Cb2W_pose"]
    T_g_c = invert_transform(data["T_C2Cb"])

    timestamps, T_w_g = tum_poses_to_transforms(poses)
    T_w_c = np.array([T_w_g[i] @ T_g_c for i in range(len(T_w_g))])
    return timestamps, T_w_c


# =========================
# OpenCV Drawing
# =========================

def draw_axes(image, K, T):
    axis = np.array([[0,0,0],[0.1,0,0],[0,0.1,0],[0,0,0.1]])

    rvec, _ = cv2.Rodrigues(T[:3,:3])
    tvec = T[:3,3]

    pts, _ = cv2.projectPoints(axis, rvec, tvec, K, None)
    pts = pts.reshape(-1,2)

    for i, c in enumerate(AXIS_COLORS):
        cv2.line(image, tuple(pts[0].astype(int)), tuple(pts[i+1].astype(int)), c, 3)

    return image


def plot_cube(image, K, T, scale, color):
    verts = np.array([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],
        [0,0,1],[1,0,1],[1,1,1],[0,1,1]
    ])

    verts = verts * 2 - 1
    verts = verts @ scale.T

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    rvec, _ = cv2.Rodrigues(T[:3,:3])
    tvec = T[:3,3]

    for i,j in edges:
        pts = np.array([verts[i], verts[j]])
        proj, _ = cv2.projectPoints(pts, rvec, tvec, K, None)
        p1, p2 = proj.reshape(2,2)
        cv2.line(image, tuple(p1.astype(int)), tuple(p2.astype(int)), color, 2)

    return image


def project_trajectory(image, traj, K, color):
    if len(traj) < 2:
        return image

    pts = np.array(traj)
    proj, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, None)
    proj = proj.reshape(-1,2)

    for i in range(1, len(proj)):
        cv2.line(image,
                 tuple(proj[i-1].astype(int)),
                 tuple(proj[i].astype(int)),
                 color, 2)
    return image


# =========================
# 3D Plot
# =========================

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

def plot_trajectories(T_w_c, T_w_o_gt, T_w_o_mocap):
    traj_cam = T_w_c[:, :3, 3]
    traj_object_gt = T_w_o_gt[:, :3, 3]
    traj_object_mocap = T_w_o_mocap[:, :3, 3]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(traj_cam[:,0], traj_cam[:,1], traj_cam[:,2], label="Camera", color='blue')
    ax.plot(traj_object_gt[:,0], traj_object_gt[:,1], traj_object_gt[:,2], label="Object GT", color='green')
    ax.plot(traj_object_mocap[:,0], traj_object_mocap[:,1], traj_object_mocap[:,2], label="Object MOCAP", color='red')

    ax.legend()
    ax.set_title("Trajectories (World Frame)")


    set_axes_equal(ax)
    plt.show()


def write_pose_csv_gtsam(file_path, timestamps, T_w_x):
    import csv
    import gtsam
    """
    Write poses to CSV using GTSAM Pose3:
    timestamp, tx, ty, tz, qx, qy, qz, qw
    """
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

        for i in range(T_w_x.shape[0]):
            pose = gtsam.Pose3(T_w_x[i])

            t = pose.translation()
            q = pose.rotation().toQuaternion()  # gtsam.Quaternion

            writer.writerow([
                timestamps[i],
                t[0], t[1], t[2],
                q.x(), q.y(), q.z(), q.w()
            ])

# =========================
# Main
# =========================

def main(args):
    data_dir = Path(args.data_dir)
    seq = data_dir.name

    fx, fy, cx, cy, K, D = load_intrinsics(data_dir / "intrinsics.txt")

    pose_path = Path(args.pose_dir) / f"{seq}_o2c_pose.npz"
    poses = np.load(pose_path)

    timestamps_gt, T_o_c_gt = tum_poses_to_transforms(poses["gt"])
    timestamps_mocap, T_o_c_mocap = tum_poses_to_transforms(poses["mocap"])

    cam_pose_file = data_dir / "cam_annotations" / f"{seq}_cam_pose.npz"
    timestamps_c, T_w_c = load_camera_poses(cam_pose_file)


    T_w_c_traj = PoseTrajectory3D(poses_se3=T_w_c, timestamps=timestamps_c)
    T_o_c_gt_traj = PoseTrajectory3D(poses_se3=T_o_c_gt, timestamps=timestamps_gt )
    T_o_c_mocap_traj = PoseTrajectory3D(poses_se3=T_o_c_mocap, timestamps=timestamps_mocap)

    T_o_c_gt_traj_sync, T_o_c_mocap_traj_sync = associate_trajectories(
        T_o_c_gt_traj, T_o_c_mocap_traj, max_diff=0.01
    )

    T_w_c_traj_sync, _ = associate_trajectories(
        T_w_c_traj, T_o_c_mocap_traj_sync, max_diff=0.01
    )

    T_o_c_gt_traj = np.array(T_o_c_gt_traj_sync.poses_se3)
    T_o_c_mocap = np.array(T_o_c_mocap_traj_sync.poses_se3)
    T_w_c = np.array(T_w_c_traj_sync.poses_se3)

    T_w_o_gt = np.array([T_w_c[i] @ T_o_c_gt[i] for i in range(T_o_c_gt.shape[0])])
    T_w_o_mocap = np.array([T_w_c[i] @ T_o_c_mocap[i] for i in range(T_o_c_mocap.shape[0])])

    timestamps_gt = np.array(T_o_c_gt_traj_sync.timestamps)
    timestamps_mocap = np.array(T_o_c_mocap_traj_sync.timestamps)
    timestamps_c = np.array(T_w_c_traj_sync.timestamps)

    print(timestamps_c)

    assert T_w_o_gt.shape[0] == timestamps_c.shape[0], f"{T_w_o_gt.shape[0]} vs {timestamps_c.shape[0]}"
    assert T_w_o_mocap.shape[0] == timestamps_c.shape[0]

    images = sorted((data_dir / "color").glob("*_color.png"))

    assert T_w_c.shape[0] == len(images)

    model_path = resolve_model_path(data_dir)
    scale = get_model_scale(model_path)


    if args.show_overlap:
        for i, img_file in enumerate(images):
            render = cv2.imread(img_file)
            if render is None:
                continue
            # img = cv2.resize(img, (W_img, H_img))

            T_c_w = T_w_c[i]
            T_w_c_inv = invert_transform(T_c_w)

            T_c_o_gt_i = T_w_c_inv @ T_w_o_gt[i]
            T_c_o_mocap_i = T_w_c_inv @ T_w_o_mocap[i]

            render = draw_axes(render, K, T_c_o_gt_i)
            render = plot_cube(render, K, T_c_o_gt_i, scale, GT_BBOX_COLOR)

            render = draw_axes(render, K, T_c_o_mocap_i)
            render = plot_cube(render, K, T_c_o_mocap_i, scale, MOCAP_BBOX_COLOR)

            # render = project_trajectory(render, cam_traj, K, (0,255,255))

            cv2.imshow("Camera + Object Axes", render)
            key = cv2.waitKey(50)
            if key == 27:
                break

        cv2.destroyAllWindows()

    print("Plotting trajectories...")
    plot_trajectories(T_w_c, T_w_o_gt, T_w_o_mocap)

    if args.save_trajectories:
        # now write ground truth camera pose and object pose to file
        # in csv format with timestamp, tx, ty, tz, qx, qy,qz, qw
        camera_csv = data_dir / "camera_poses.csv"
        object_csv = data_dir / "object_poses.csv"
        print(f"Writing camera and object gt csv files {camera_csv} {object_csv}")

        write_pose_csv_gtsam(camera_csv, timestamps_gt, T_w_c)
        write_pose_csv_gtsam(object_csv, timestamps_gt, T_w_o_gt)


# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data_dir", required=True)
    parser.add_argument("--pose_dir", default=str(PROJECT_ROOT / "VAL10Pose"))
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--show_overlap", action='store_true')
    parser.add_argument("-s", "--save_trajectories", action='store_true')

    args = parser.parse_args()

    main(args)
