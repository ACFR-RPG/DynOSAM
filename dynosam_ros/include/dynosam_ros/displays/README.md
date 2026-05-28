# Dynamic SLAM Displays

Version of the front/back-end displays that depends on `dynamic_slam_interfaces` to publish the object states per frame:

- object pose
- object paths
- object motions
- object velocities
- object id

Each object is represented using the `dynamic_slam_interfaces::msg::ObjectOdometry` message which can be displayed in RVIZ using the [rviz_dynamic_slam_plugins](https://github.com/ACFR-RPG/rviz_dynamic_slam_plugins) plugin.
