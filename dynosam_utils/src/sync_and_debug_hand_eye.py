#!/usr/bin/env python3
import os
import sqlite3
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

# ROS 2 Deserialization imports
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge

# ==========================================
# CONFIGURATION PARAMETERS
# ==========================================
BAG_PATH = "path/to/your/rosbag_directory"  # Directory containing metadata.yaml and .db3
IMAGE_TOPIC = "/camera/image_raw"
POSE_TOPIC = "/vicon/camera_marker/pose"    # Expected: geometry_msgs/msg/PoseStamped

# Checkerboard Settings
CHECKERBOARD_SIZE = (9, 6)   # Inner corners (width, height)
SQUARE_SIZE = 0.025          # Size of a square side in meters

# Synchronization Threshold
MAX_TIME_DIFF_NS = 30_000_000  # 30 milliseconds max delta
# ==========================================

class BagExtractor:
    def __init__(self, bag_path):
        self.bag_path = bag_path
        self.bridge = CvBridge()

        # Discover the SQLite3 database file inside the bag directory
        self.db_file = None
        for file in os.listdir(bag_path):
            if file.endswith('.db3'):
                self.db_file = os.path.join(bag_path, file)
                break
        if not self.db_file:
            raise FileNotFoundError(f"No .db3 database file found in {bag_path}")

        # Connect to SQLite to read raw messages
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()

        # Map topic names to IDs and types
        self.topic_info = {}
        self.cursor.execute("SELECT id, name, type FROM topics")
        for row in self.cursor.fetchall():
            self.topic_info[row[1]] = {"id": row[0], "type": row[2]}

    def get_messages(self, topic_name):
        """Yields (timestamp_ns, msg) for a given topic."""
        if topic_name not in self.topic_info:
            print(f"Warning: Topic {topic_name} not found in bag.")
            return

        topic_id = self.topic_info[topic_name]["id"]
        msg_type_str = self.topic_info[topic_name]["type"]
        msg_type = get_message(msg_type_str)

        self.cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC",
            (topic_id,)
        )

        for timestamp, raw_data in self.cursor.fetchall():
            msg = deserialize_message(raw_data, msg_type)
            yield timestamp, msg

    def close(self):
        self.conn.close()


def pose_to_matrix(pose_msg):
    """Converts a geometry_msgs/Pose into a 4x4 Homogeneous Transformation Matrix."""
    pos = pose_msg.position
    ori = pose_msg.orientation

    T = np.eye(4)
    T[0:3, 3] = [pos.x, pos.y, pos.z]
    r = R.from_quat([ori.x, ori.y, ori.z, ori.w])
    T[0:3, 0:3] = r.as_matrix()
    return T


def sync_data(bag_extractor, img_topic, pose_topic, max_diff_ns):
    """Greedily synchronizes images and poses based on closest timestamps."""
    print("Extracting data from bag...")
    images = list(bag_extractor.get_messages(img_topic))
    poses = list(bag_extractor.get_messages(pose_topic))

    print(f"Loaded {len(images)} images and {len(poses)} poses. Syncing...")

    synced_pairs = []
    pose_idx = 0

    for img_ts, img_msg in images:
        while pose_idx < len(poses) - 1 and abs(poses[pose_idx+1][0] - img_ts) < abs(poses[pose_idx][0] - img_ts):
            pose_idx += 1

        pose_ts, pose_msg = poses[pose_idx]
        time_diff = abs(img_ts - pose_ts)

        if time_diff <= max_diff_ns:
            synced_pairs.append({
                'img_msg': img_msg,
                'pose_msg': pose_msg,
                'time_diff_ms': time_diff / 1e6
            })

    print(f"Successfully synchronized {len(synced_pairs)} frame-pose pairs.")
    return synced_pairs


def run_hand_eye_calibration(valid_obj_points, valid_corners, valid_poses_B, image_shape):
    """
    Performs camera intrinsic calibration followed by Eye-in-Hand Calibration.

    Mathematical convention handled here:
    - We solve for the static transformation T_B_A (From Camera Optical Frame A to Vicon Frame B).
    - OpenCV's calibrateHandEye takes:
        R_gripper2base (R_Wv_B): Rotation from Gripper(B) to Base(Wv)
        t_gripper2base (t_Wv_B): Translation from Gripper(B) to Base(Wv)
        R_target2cam   (R_A_C):  Rotation from Camera(A) to Target/Checkerboard(C)
        t_target2cam   (t_A_C):  Translation from Camera(A) to Target/Checkerboard(C)
    """
    print("\n==============================================")
    print("RUNNING HAND-EYE CALIBRATION")
    print("==============================================")

    # 1. Estimate Camera Intrinsics using the collected checkerboard frames
    print("Step 1: Computing camera intrinsics via OpenCV calibrateCamera...")
    ret, K, dist, rvecs_A_C, tvecs_A_C = cv2.calibrateCamera(
        valid_obj_points, valid_corners, image_shape[::-1], None, None
    )

    if not ret:
        print("[Error] Intrinsic camera calibration failed.")
        return

    print("Camera Intrinsic Matrix (K):\n", K)
    print("Distortion Coefficients:\n", dist.ravel())

    # 2. Re-arrange Vicon poses and Camera-Target poses for OpenCV's solver
    R_base_gripper = []  # R_Wv_B
    t_base_gripper = []  # t_Wv_B
    R_target_cam = []    # R_A_C
    t_target_cam = []    # t_A_C

    for i in range(len(valid_poses_B)):
        # Vicon Pose: T_Wv_B (Transform from B to Vicon World)
        T_Wv_B = valid_poses_B[i]
        R_base_gripper.append(T_Wv_B[0:3, 0:3])
        t_base_gripper.append(T_Wv_B[0:3, 3])

        # Camera Target Pose: OpenCV provides rvec/tvec from Target(C) to Camera(A) -> T_A_C
        # Extracted directly from calibrateCamera outputs
        R_A_C, _ = cv2.Rodrigues(rvecs_A_C[i])
        R_target_cam.append(R_A_C)
        t_target_cam.append(tvecs_A_C[i])

    # 3. Solve Hand-Eye Calibration (AX = XB variant)
    print("\nStep 2: Solving Hand-Eye calibration equations...")
    # Using Tsai-Lenz method (TSAI) as it is highly robust for standard trajectory configurations
    R_B_A, t_B_A = cv2.calibrateHandEye(
        R_base_gripper, t_base_gripper,
        R_target_cam, t_target_cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # Construct final transformation matrix T_B_A (Camera Frame A expressed in Vicon Frame B)
    T_B_A = np.eye(4)
    T_B_A[0:3, 0:3] = R_B_A
    T_B_A[0:3, 3] = t_B_A.ravel()

    print("\n================ CALIBRATION RESULT ================")
    print("Transformation Matrix from Camera Optical (A) to Vicon (B) [T_B_A]:")
    print(np.array2string(T_B_A, formatter={'float_kind': lambda x: f"{x:10.6f}"}))

    # Also print out inverse transformation T_A_B for flexibility
    T_A_B = np.linalg.inv(T_B_A)
    print("\nTransformation Matrix from Vicon (B) to Camera Optical (A) [T_A_B]:")
    print(np.array2string(T_A_B, formatter={'float_kind': lambda x: f"{x:10.6f}"}))
    print("====================================================")


def main():
    extractor = BagExtractor(BAG_PATH)
    synced_data = sync_data(extractor, IMAGE_TOPIC, POSE_TOPIC, MAX_TIME_DIFF_NS)

    # 3D object points for checkerboard (Z=0 plane)
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    valid_poses_B = []
    valid_corners = []
    valid_obj_points = []
    img_shape = None

    print("\n--- Starting Visual Verification Pass ---")
    print("Press 's' to ACCEPT a frame for final calibration.")
    print("Press 'd' to REJECT / skip a frame.")
    print("Press 'q' to EXIT the sequence.")

    cv2.namedWindow("Calibration Debug Stream", cv2.WINDOW_NORMAL)

    for idx, data in enumerate(synced_data):
        cv_img = extractor.bridge.imgmsg_to_cv2(data['img_msg'], desired_encoding="bgr8")
        if img_shape is None:
            img_shape = cv_img.shape[:2] # (height, width)

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

        debug_img = cv_img.copy()
        status_text = f"Frame {idx+1}/{len(synced_data)} | Sync Diff: {data['time_diff_ms']:.1f}ms"
        cv2.putText(debug_img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(debug_img, CHECKERBOARD_SIZE, corners_refined, ret)
            cv2.putText(debug_img, "CHESSBOARD DETECTED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(debug_img, "CHESSBOARD NOT FOUND", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Calibration Debug Stream", debug_img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s') and ret:
            T_Wv_B = pose_to_matrix(data['pose_msg'].pose)
            valid_poses_B.append(T_Wv_B)
            valid_corners.append(corners_refined)
            valid_obj_points.append(objp)
            print(f"-> Frame {idx+1} accepted. ({len(valid_poses_B)} total accepted frames)")

        elif key == ord('d'):
            print(f"-> Frame {idx+1} skipped by user.")
            continue
        elif key == ord('q'):
            print("Exiting sequence analysis.")
            break

    cv2.destroyAllWindows()
    extractor.close()

    if len(valid_poses_B) < 3:
        print("\n[Error] Not enough frames accepted. Hand-eye calibration requires at least 3 distinct non-parallel poses.")
        return

    print(f"\nCollected {len(valid_poses_B)} valid samples.")

    # =========================================================================
    # UNCOMMENT THE LINE BELOW TO EXECUTE THE CALIBRATION ROUTINE
    # =========================================================================
    # run_hand_eye_calibration(valid_obj_points, valid_corners, valid_poses_B, img_shape)

if __name__ == "__main__":
    main()
