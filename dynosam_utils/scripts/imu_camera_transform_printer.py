import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import gtsam
import numpy as np

def estimate_initial_state_and_bias(imu_samples):
    """
    Estimates initial orientation and biases from stationary IMU data,
    assuming a standard Robotics world frame (X-Forward, Y-Left, Z-Up).

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

    # 3. Known or prior accelerometer bias from calibration tool
    init_accel_bias = np.array([-0.029419409374627146, 0.13233094180247623, -0.04503877162618777])

    # 4. Correct the mean accelerometer reading using prior bias BEFORE orientation math
    corrected_accel = mean_accel - init_accel_bias
    v_body = corrected_accel / np.linalg.norm(corrected_accel)

    # In Z-Up coordinates, the static upward acceleration vector aligns with +Z
    v_world = np.array([0.0, 0.0, 1.0])

    # Cross product yields the orthogonal axis of rotation
    axis = np.cross(v_body, v_world)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        if np.dot(v_body, v_world) > 0:
            init_rotation = gtsam.Rot3.Identity()
        else:
            # 180-degree flip around X-axis if completely upside down
            init_rotation = gtsam.Rot3.Rx(np.pi)
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(v_body, v_world), -1.0, 1.0))
        rot_vector = axis * angle
        init_rotation = gtsam.Rot3.Expmap(rot_vector)

    prior_bias = gtsam.imuBias.ConstantBias(init_accel_bias, init_gyro_bias)
    return init_rotation, prior_bias

class ImuPreintegrationNode(Node):
    def __init__(self):
        super().__init__('imu_preintegration_node')

        self.is_initialized = False
        self.init_samples = []
        self.samples_required = 400 # At 200Hz, this is ~2 seconds of stationary data

        # Subscriptions and Publishers
        self.imu_sub = self.create_subscription(Imu, '/d455/imu', self.imu_callback, qos_profile_sensor_data)
        self.path_pub = self.create_publisher(Path, '/imu/trajectory', 10)

        # Variance setup
        import math
        measured_acc_variance = np.eye(3) * math.pow(0.0010705806893328113, 2)
        measured_omega_variance = np.eye(3) * math.pow(0.00020102490399477883, 2)
        integration_variance = np.eye(3) * math.pow(0.1, 2)

        # Extract the absolute gravity magnitude
        # kalibr_g_vector = np.array([-0.03007675, -9.80262369,  0.27583877])
        # kalibr_g_vector = np.array([0, -9.80262369,  ])
        # g_magnitude = np.linalg.norm(kalibr_g_vector)

        # ROBOTICS CONVENTION: Gravity points straight down along negative Z
        gravity_world = np.array([0.0, 0.0, -9.8])
        self.get_logger().info(f"Configuring GTSAM World Gravity Vector (Z-UP) as: {gravity_world}")

        params = gtsam.PreintegrationParams(gravity_world)
        params.setAccelerometerCovariance(measured_acc_variance)
        params.setGyroscopeCovariance(measured_omega_variance)
        params.setIntegrationCovariance(integration_variance)

        # Initialize temporary zero bias state
        self.prior_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        self.pim = gtsam.PreintegratedImuMeasurements(params, self.prior_bias)

        self.current_state = gtsam.NavState(gtsam.Pose3(), np.zeros(3))

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'
        self.last_timestamp = None
        self.get_logger().info("GTSAM IMU Preintegration Node started.")

    def imu_callback(self, msg: Imu):
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
                init_rot, self.prior_bias = estimate_initial_state_and_bias(self.init_samples)

                # Set initial state orientation with zero initial position and velocity
                self.current_state = gtsam.NavState(gtsam.Pose3(init_rot, gtsam.Point3(0, 0, 0)), np.zeros(3))

                # Update PIM instance with newly evaluated real bias parameters
                self.pim.resetIntegrationAndSetBias(self.prior_bias)

                self.last_timestamp = current_timestamp
                self.is_initialized = True
                self.get_logger().info("IMU Orientation and Bias Initialized successfully!")
            return
        print(f"Accel={accel} Gyro={gyro}")
        # 1. Integrate measurement
        self.pim.integrateMeasurement(accel, gyro, dt)

        # 2. Forward predict state mapping
        predicted_state = self.pim.predict(self.current_state, self.prior_bias)

        # Monitor stability: velocity vectors should hover around [0.0, 0.0, 0.0]
        print(f"Current Velocity State: {predicted_state.velocity()}")

        # Update state wrappers
        self.current_state = predicted_state
        self.last_timestamp = current_timestamp

        # 3. Clean accumulator for the next interval step
        self.pim.resetIntegration()

        # 4. Fire to visualization path
        self.publish_trajectory(msg.header.stamp, predicted_state.pose())

    def publish_trajectory(self, stamp, pose):
        position = pose.translation()
        quat = pose.rotation().toQuaternion()

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = 'odom'

        pose_stamped.pose.position.x = position[0]
        pose_stamped.pose.position.y = position[1]
        pose_stamped.pose.position.z = position[2]

        pose_stamped.pose.orientation.x = quat.x()
        pose_stamped.pose.orientation.y = quat.y()
        pose_stamped.pose.orientation.z = quat.z()
        pose_stamped.pose.orientation.w = quat.w()

        self.path_msg.poses.append(pose_stamped)
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
