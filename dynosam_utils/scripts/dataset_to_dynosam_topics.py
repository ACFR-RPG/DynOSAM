#!/usr/bin/env python3

import os
import glob
import argparse
import cv2
import threading
import sys
import termios
import tty
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


def print_progress_bar(current, total, length=40):
    """
    Prints a dynamic progress bar in the terminal.

    current : int - current frame index
    total   : int - total number of frames
    length  : int - number of characters for the bar
    """
    percent = current / total
    filled_len = int(length * percent)
    bar = '#' * filled_len + '-' * (length - filled_len)
    print(f"\r[{bar}] {percent*100:6.2f}%", end='', flush=True)
    if current >= total - 1:
        print()  # newline at end

class DynosamDatasetPublisher(Node):
    def __init__(self, args):
        super().__init__('dynosam_dataset_publisher')

        self.bridge = CvBridge()
        self.folder = args.folder_path

        self.publish_rgb = args.rgb
        self.publish_depth = args.depth
        self.publish_mask = args.mask
        self.intrinsics_file = args.intrinsics_file

        self.loop = args.loop
        self.paused = args.paused
        self.publish_rate = args.rate

        # Publishers
        if self.publish_rgb:
            self.rgb_pub = self.create_publisher(Image, '/camera/rgb', 10)
        if self.publish_depth:
            self.depth_pub = self.create_publisher(Image, '/camera/depth', 10)
        if self.publish_mask:
            self.mask_pub = self.create_publisher(Image, '/camera/mask', 10)

        self.cam_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        # --------- Load and validate files ---------
        try:
            self.rgb_files = self.load_files_checked('rgb') if self.publish_rgb else None
            self.depth_files = self.load_files_checked('depth') if self.publish_depth else None
            self.mask_files = self.load_files_checked('masks') if self.publish_mask else None

            self.validate_file_lists()
            self.check_image_shapes()

        except RuntimeError as e:
            self.get_logger().fatal(str(e))
            raise  # stop node immediately

        self.num_frames = len(
            self.rgb_files or self.depth_files or self.mask_files
        )
        self.get_logger().info(f"Loaded {self.num_frames} frames")

        # Load intrinsics (auto-detect K matrix or fx/fy/cx/cy)
        first_image_shape = cv2.imread(self.rgb_files[0], cv2.IMREAD_UNCHANGED).shape \
            if self.rgb_files else None
        self.camera_info_msg = self.load_intrinsics(self.intrinsics_file, first_image_shape)

        self.idx = 0

        # Timer according to user rate
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        # Keyboard thread
        self.stop_keyboard_thread = threading.Event()
        self.key_thread = threading.Thread(target=self.keyboard_listener)
        self.key_thread.start()

        if self.paused:
            self.get_logger().info("Starting in PAUSED state. Press SPACE to resume.")

    def destroy_node(self):
        self.get_logger().info("Destroying node...")
        # Signal thread to stop and join before destroying node
        self.stop_keyboard_thread.set()
        self.key_thread.join()
        super().destroy_node()

    # --------- File loading + validation ---------
    def load_files_checked(self, subfolder):
        path = os.path.join(self.folder, subfolder)

        if not os.path.exists(path):
            raise RuntimeError(f"Required folder '{subfolder}' not found at: {path}")

        files = sorted(glob.glob(os.path.join(path, '*')))

        if len(files) == 0:
            raise RuntimeError(f"No files found in required folder: {path}")

        self.get_logger().info(f"{subfolder}: {len(files)} files")

        return files

    def validate_file_lists(self):
        lengths = {}

        if self.rgb_files is not None:
            lengths['rgb'] = len(self.rgb_files)
        if self.depth_files is not None:
            lengths['depth'] = len(self.depth_files)
        if self.mask_files is not None:
            lengths['mask'] = len(self.mask_files)

        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            msg = "Mismatch in number of files across modalities:\n"
            for k, v in lengths.items():
                msg += f"  {k}: {v}\n"
            raise RuntimeError(msg.strip())

    def check_image_shapes(self):
        """Ensure all images have the same dimensions"""
        image_lists = [l for l in [self.rgb_files, self.depth_files, self.mask_files] if l]
        if not image_lists:
            return

        first_shape = cv2.imread(image_lists[0][0], cv2.IMREAD_UNCHANGED).shape
        for images in image_lists:
            for i, f in enumerate(images):
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                # only check width and height of shape not number of channels
                if img.shape[:2] != first_shape[:2]:
                    raise RuntimeError(
                        f"Image mismatch at {f}: {img.shape}, expected {first_shape}"
                    )

    # --------- Intrinsics ---------
    def load_intrinsics(self, intrinsics_file_name, image_shape=None):
        intrinsics_path = os.path.join(self.folder, intrinsics_file_name)
        self.get_logger().info(f"Loading intrinsics: {intrinsics_file_name}")

        if not os.path.exists(intrinsics_path):
            raise RuntimeError(f"Intrinsics file not found at: {intrinsics_path}")

        vals = np.loadtxt(intrinsics_path)
        cam_info = CameraInfo()
        cam_info.distortion_model = "plumb_bob"
        cam_info.d = [0.0] * 5

        # Detect full K matrix
        if vals.shape == (3, 3):
            fx, fy = vals[0, 0], vals[1, 1]
            cx, cy = vals[0, 2], vals[1, 2]
            width, height = image_shape[1], image_shape[0] if image_shape else 0
            self.get_logger().info(f"Loaded full K matrix: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        elif vals.size == 4:
            fx, fy, cx, cy = vals.flatten()
            width, height = image_shape[1], image_shape[0] if image_shape else 0
            self.get_logger().info(f"Loaded intrinsics vector fx,fy,cx,cy")
        else:
            raise RuntimeError(
                f"Unsupported intrinsics format: {vals.shape}. Provide 3x3 K or fx fy cx cy"
            )

        cam_info.width = int(width)
        cam_info.height = int(height)

        cam_info.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]

        cam_info.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        return cam_info

    def load_rgb(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.get_logger().warn(f"Failed to load {image_path}")
            return None, None

        return img, "bgr8"

    def load_depth(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            self.get_logger().warn(f"Failed to load {image_path}")
            return None, None

        return img, "mono16"

    def load_mask(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.get_logger().warn(f"Failed to load {image_path}")
            return None, None


        return img.astype(np.int32), "32SC1"

    # --------- Publishing ---------
    def publish_image(self, pub, image, encoding, stamp):
        if image is None:
            self.get_logger().warn(f"Failed to load publish image")
            return

        msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
        msg.header.stamp = stamp
        pub.publish(msg)

    def timer_callback(self):
        if self.paused:
            return

        if self.idx >= self.num_frames:
            if self.loop:
                self.get_logger().info("Looping sequence...")
                self.idx = 0
            else:
                self.get_logger().info("Finished sequence.")
                # Show full progress bar at the end
                print_progress_bar(self.num_frames, self.num_frames)

                # Stop the timer so callback is no longer called
                self.timer.cancel()

                # Signal keyboard thread to exit
                self.stop_keyboard_thread.set()
                self.key_thread.join()

                # Shutdown ROS cleanly
                rclpy.shutdown()
                return

        stamp = self.get_clock().now().to_msg()

        # Publish images
        if self.publish_rgb:
            image, encoding = self.load_rgb(self.rgb_files[self.idx])
            self.publish_image(self.rgb_pub, image, encoding, stamp)
        if self.publish_depth:
            image, encoding = self.load_depth(self.depth_files[self.idx])
            self.publish_image(self.depth_pub, image, encoding, stamp)
        if self.publish_mask:
            image, encoding = self.load_mask(self.mask_files[self.idx])
            self.publish_image(self.mask_pub, image, encoding, stamp)

        # Publish camera info
        self.camera_info_msg.header.stamp = stamp
        self.cam_info_pub.publish(self.camera_info_msg)

        # Update terminal progress bar
        print_progress_bar(self.idx + 1, self.num_frames)

        self.idx += 1

    # --------- Keyboard control ---------
    def keyboard_listener(self):
        import select
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            while rclpy.ok() and not self.stop_keyboard_thread.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == ' ':
                        self.paused = not self.paused
                        state = "PAUSED" if self.paused else "RUNNING"
                        self.get_logger().info(f"Toggled playback: {state}")
        finally:
            # Always restore terminal
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class RSLObjectGraspingLoader(DynosamDatasetPublisher):

    def __init__(self, args):
        super().__init__(args)

    def load_mask(self, mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # rsl grsping dataset provides binary mask with background 0 and object 1 (ie 255)
        # convert to background = 0 and object is 1
        # Ensure it's binary: 0 for background, 255 for object
        # Optional thresholding if the image is not strictly 0/255
        _, binary_img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Convert to 32-bit signed integer, scale object pixels to 1
        return (binary_img // 255).astype(np.int32), "32SC1"  # background=0, object=1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder_path', required=True)
    parser.add_argument(
        '-i',
        '--intrinsics_file',
        default='intrinsics.txt',
        help='Name of intrinsics file inside folder_path (default: intrinsics.txt)'
    )
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--depth', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('-l', '--loop', action='store_true', help='Loop sequence when finished')
    parser.add_argument('-p', '--paused', action='store_true', help='Start in paused state')
    parser.add_argument(
        '-r', '--rate', type=float, default=10.0,
        help='Publishing rate in Hz (default: 10 Hz)'
    )

    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    logger = get_logger("dataset_to_dynosam_node")

    try:
        node = RSLObjectGraspingLoader(args)
        # node = DynosamDatasetPublisher(args)
    except RuntimeError as e:
        logger.fatal(str(e))
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()  # joins the keyboard thread
        rclpy.shutdown()

    # node.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()
