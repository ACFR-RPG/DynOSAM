# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node
# import tf2_ros
# import numpy as np

# from scipy.spatial.transform import Rotation as R

# class TfDumper(Node):
#     def __init__(self):
#         super().__init__('tf_dumper')
#         self.buffer = tf2_ros.Buffer()
#         self.listener = tf2_ros.TransformListener(self.buffer, self)

#     def dump(self):
#         tf = self.buffer.lookup_transform(
#             'camera_color_optical_frame',
#             'camera_imu_optical_frame',
#             rclpy.time.Time()
#         )

#         t = tf.transform.translation
#         q = tf.transform.rotation

#         # YAML-style output (quiet, copy-paste friendly)
#         # print("transform:")
#         # print(f"  translation: [{t.x:.6f}, {t.y:.6f}, {t.z:.6f}]")
#         # print(f"  rotation: [{q.x:.6f}, {q.y:.6f}, {q.z:.6f}, {q.w:.6f}]")

#         t = tf.transform.translation
#         q = tf.transform.rotation

#         # SciPy expects [x, y, z, w]
#         rot = R.from_quat([q.x, q.y, q.z, q.w])
#         R_mat = rot.as_matrix()

#         T = np.eye(4)
#         T[:3, :3] = R_mat
#         T[:3, 3] = [t.x, t.y, t.z]

#         print("T_imu_to_optical:")
#         for row in T:
#             print("  - [" + ", ".join(f"{v:.6f}" for v in row) + "]")


#         # except Exception:
#         #     pass  # stay quiet


# def main():
#     rclpy.init()
#     node = TfDumper()

#     # give TF a moment to fill
#     rclpy.spin_once(node, timeout_sec=4.0)

#     node.dump()

#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_default, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import gtsam
import numpy as np

def estimate_initial_state_and_bias(imu_samples):
    """
    Estimates initial orientation and biases from stationary IMU data,
    assuming an OpenCV world frame (X-Right, Y-Down, Z-Forward).

    :param imu_samples: List of dicts containing 'accel' and 'gyro' as np.arrays
    :return: tuple (gtsam.Rot3, gtsam.imuBias.ConstantBias)
    """
    if len(imu_samples) == 0:
        raise ValueError("Sample list is empty. Cannot initialize.")

    # 1. Average the raw measurements
    mean_accel = np.mean([s['accel'] for s in imu_samples], axis=0)
    mean_gyro = np.mean([s['gyro'] for s in imu_samples], axis=0)

    # 2. Gyroscope average maps to the initial gyroscope bias
    init_gyro_bias = mean_gyro

    # 3. Compute rotation aligning measured gravity with World Y [0, 1, 0]
    v_body = mean_accel / np.linalg.norm(mean_accel)

    # In OpenCV coordinates, gravity points directly down along +Y
    v_world = np.array([0.0, 1.0, 0.0])

    # Cross product yields the orthogonal axis of rotation
    axis = np.cross(v_body, v_world)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        # Vectors are already aligned or exactly inverted
        if np.dot(v_body, v_world) > 0:
            init_rotation = gtsam.Rot3.Identity()
        else:
            # 180-degree flip around Z-axis if completely upside down
            init_rotation = gtsam.Rot3.Rz(np.pi)
    else:
        # Normalize the axis and extract the angle via dot product
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(v_body, v_world), -1.0, 1.0))

        # Build the rotation vector and map it using Expmap
        rot_vector = axis * angle
        init_rotation = gtsam.Rot3.Expmap(rot_vector)

    # 4. Construct ConstantBias (assuming 0 initial accelerometer bias)
    init_accel_bias =  np.array([-0.029419409374627146,0.13233094180247623,-0.04503877162618777])
    prior_bias = gtsam.imuBias.ConstantBias(init_accel_bias, init_gyro_bias)

    return init_rotation, prior_bias

class ImuPreintegrationNode(Node):
    def __init__(self):
        super().__init__('imu_preintegration_node')

        # ... your existing setup ...
        self.is_initialized = False
        self.init_samples = []
        self.samples_required = 200 # At 100Hz, this is ~2 seconds of stationary data

        # Subscriptions and Publishers
        self.imu_sub = self.create_subscription(Imu, '/d455/imu', self.imu_callback, qos_profile_sensor_data)
        self.path_pub = self.create_publisher(Path, '/imu/trajectory', 10)

        # Initialize GTSAM Preintegration Parameters
        # Assuming a standard IMU (accel noise, gyro noise, integration noise)
        # In a production environment, pull these from a YAML calibration file.
        measured_acc_variance = np.eye(3) * 0.4
        measured_omega_variance = np.eye(3) * 0.4
        integration_variance = np.eye(3) * 0.1

        # params = gtsam.PreintegrationParams.MakeSharedU(9.81) # Gravity along Z
        params = gtsam.PreintegrationParams(np.array([0, -9.81, 0]))
        params.setAccelerometerCovariance(measured_acc_variance)
        params.setGyroscopeCovariance(measured_omega_variance)
        params.setIntegrationCovariance(integration_variance)
        params.setUse2ndOrderCoriolis(True)

        # Initialize Bias (assuming zero initial bias)
        self.prior_bias = gtsam.imuBias.ConstantBias(
            np.array([-0.029419409374627146,0.13233094180247623,-0.04503877162618777]),
            np.array([-0.001957312546791094, -0.002247023068157285,0.00010864144551232155])
        )

        # Preintegrator objects
        self.pim = gtsam.PreintegratedImuMeasurements(params, self.prior_bias)

        # Current State Tracking (NavState holds Pose3 and Velocity3)
        self.current_state = gtsam.NavState(gtsam.Pose3(), np.zeros(3))

        # Trajectory Message Configuration
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'

        self.last_timestamp = None
        self.get_logger().info("GTSAM IMU Preintegration Node started.")

    def imu_callback(self, msg: Imu):
        # print(msg)
        # Extract timestamp in seconds
        current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_timestamp is None:
            self.last_timestamp = current_timestamp
            return

        dt = current_timestamp - self.last_timestamp
        if dt <= 0.0:
            return

        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        # Phase 1: Initialization Loop
        if not self.is_initialized:
            self.init_samples.append({'accel': accel, 'gyro': gyro})

            if len(self.init_samples) >= self.samples_required:
                # Run estimation routine
                init_rot, self.prior_bias = estimate_initial_state_and_bias(self.init_samples)

                print(self.prior_bias)

                # Update initial state with correct orientation alignment
                self.current_state = gtsam.NavState(gtsam.Pose3(init_rot, gtsam.Point3(0,0,0)), np.zeros(3))

                # Re-initialize PIM with computed bias
                self.pim.resetIntegrationAndSetBias(self.prior_bias)

                self.last_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.is_initialized = True
                self.get_logger().info("IMU Orientation and Bias Initialized successfully!")
            return

        print(accel)
        # print(gyro)

        # 1. Integrate the current measurement
        self.pim.integrateMeasurement(accel, gyro, dt)

        # 2. Predict the new state using the accumulated preintegrated measurements
        # In a full SLAM system, you'd extract the PIM at keyframes and reset it.
        # For a simple continuous prediction, we predict forward from our last state.
        predicted_state = self.pim.predict(self.current_state, self.prior_bias)

        # self.pim.print()

        # Update current state tracker
        self.current_state = predicted_state
        self.last_timestamp = current_timestamp

        # 3. Reset the PIM if you want to avoid unbounded error propagation between steps,
        # or keep updating it depending on your factor graph design.
        # For a pure integration visualization, we reset and step forward.
        self.pim.resetIntegration()

        # 4. Publish the Trajectory
        self.publish_trajectory(msg.header.stamp, predicted_state.pose())

    def publish_trajectory(self, stamp, pose):
        position = pose.translation()
        quat = pose.rotation().toQuaternion() # Returns [w, x, y, z] in GTSAM

        # Create PoseStamped
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = 'odom'

        pose_stamped.pose.position.x = position[0]
        pose_stamped.pose.position.y = position[1]
        pose_stamped.pose.position.z = position[2]

        # ROS 2 expects [x, y, z, w] orientation
        pose_stamped.pose.orientation.x = quat.x()
        pose_stamped.pose.orientation.y = quat.y()
        pose_stamped.pose.orientation.z = quat.z()
        pose_stamped.pose.orientation.w = quat.w()

        # Append to path and limit length to avoid memory bloat
        self.path_msg.poses.append(pose_stamped)
        # if len(self.path_msg.poses) > 500:
        #     self.path_msg.poses.pop(0)

        self.path_msg.header.stamp = stamp
        self.path_pub.publish(self.path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuPreintegrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
