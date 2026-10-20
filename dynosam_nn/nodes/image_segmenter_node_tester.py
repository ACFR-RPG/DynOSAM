#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


from dynosam_nn_py import YOLODetectionEngine

class ObjectDetectioNode(Node):

    def __init__(self):
        super().__init__('object_detection_node')

        self._engine = YOLODetectionEngine()
        self.bridge = CvBridge()


        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._image_topic_callback,
            10
        )

    def _image_topic_callback(self, image: Image):
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
        result = self._engine.process(cv_image)

        viz_image = result.coloured_mask()

        cv2.imshow("Coloured Mask", viz_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)

    node = ObjectDetectioNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down node.')
    finally:
        rclpy.try_shutdown()


# To modify params run as
# run dynosam_utils record_image_topic.py --ros-args -p param:=value
if __name__ == '__main__':
    main()
