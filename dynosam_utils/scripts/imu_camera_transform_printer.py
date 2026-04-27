#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np

from scipy.spatial.transform import Rotation as R

class TfDumper(Node):
    def __init__(self):
        super().__init__('tf_dumper')
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, self)

    def dump(self):
        tf = self.buffer.lookup_transform(
            'camera_color_optical_frame',
            'camera_imu_optical_frame',
            rclpy.time.Time()
        )

        t = tf.transform.translation
        q = tf.transform.rotation

        # YAML-style output (quiet, copy-paste friendly)
        # print("transform:")
        # print(f"  translation: [{t.x:.6f}, {t.y:.6f}, {t.z:.6f}]")
        # print(f"  rotation: [{q.x:.6f}, {q.y:.6f}, {q.z:.6f}, {q.w:.6f}]")

        t = tf.transform.translation
        q = tf.transform.rotation

        # SciPy expects [x, y, z, w]
        rot = R.from_quat([q.x, q.y, q.z, q.w])
        R_mat = rot.as_matrix()

        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = [t.x, t.y, t.z]

        print("T_imu_to_optical:")
        for row in T:
            print("  - [" + ", ".join(f"{v:.6f}" for v in row) + "]")


        # except Exception:
        #     pass  # stay quiet


def main():
    rclpy.init()
    node = TfDumper()

    # give TF a moment to fill
    rclpy.spin_once(node, timeout_sec=4.0)

    node.dump()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
