#!/usr/bin/env python3
'''
Detects the red cup using depth + HSV confirmation.
Publishes the cup centroid as a PoseStamped in the sim_camera frame.
'''

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

# Camera intrinsics from /camera_info
FX = 1108.5125019853992
FY = 1108.5125019853992
CX = 640.0
CY = 360.0

# Table surface is ~0.506m from the camera.
# Objects on the table appear closer than this threshold.
TABLE_DEPTH_M    = 0.506
OBJECT_MARGIN_M  = 0.03
OBJECT_MIN_DEPTH = 0.10
OBJECT_MAX_DEPTH = TABLE_DEPTH_M - OBJECT_MARGIN_M

# HSV red ranges
RED_LOWER_1 = np.array([0,   40,  20])
RED_UPPER_1 = np.array([15,  255, 255])
RED_LOWER_2 = np.array([160, 40,  20])
RED_UPPER_2 = np.array([180, 255, 255])
MIN_RED_FRACTION  = 0.10
MIN_OBJECT_PIXELS = 200


class CupDetector(Node):
    def __init__(self):
        super().__init__("cup_detector")
        self.bridge = CvBridge()
        self._rgb   = None
        self._depth = None

        self.create_subscription(Image, "/rgb",   self._rgb_cb,   10)
        self.create_subscription(Image, "/depth", self._depth_cb, 10)

        self.pub_pose  = self.create_publisher(PoseStamped, "/red_cup_pose",  10)
        self.pub_debug = self.create_publisher(Image,       "/red_cup_debug", 10)

        self.create_timer(0.1, self._process)
        self.get_logger().info("CupDetector started")

    def _rgb_cb(self, msg):
        try:
            self._rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"RGB error: {e}")

    def _depth_cb(self, msg):
        try:
            self._depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.get_logger().warn(f"Depth error: {e}")

    def _process(self):
        if self._rgb is None or self._depth is None:
            return

        rgb   = self._rgb.copy()
        depth = self._depth.copy()

        # Find pixels above the table surface
        valid       = np.isfinite(depth)
        above_table = valid & (depth > OBJECT_MIN_DEPTH) & (depth < OBJECT_MAX_DEPTH)
        mask        = np.zeros(depth.shape, dtype=np.uint8)
        mask[above_table] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        hsv      = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1),
            cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2),
        )

        best_label = -1
        best_score = 0

        for label in range(1, n_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < MIN_OBJECT_PIXELS:
                continue
            region   = (labels == label).astype(np.uint8) * 255
            red_frac = float(np.count_nonzero(cv2.bitwise_and(red_mask, region))) / area
            if red_frac < MIN_RED_FRACTION:
                continue
            score = area * red_frac
            if score > best_score:
                best_score = score
                best_label = label

        if best_label == -1:
            self._publish_debug(rgb, mask)
            return

        cx_px = float(centroids[best_label][0])
        cy_px = float(centroids[best_label][1])

        cup_depths   = depth[labels == best_label]
        valid_depths = cup_depths[np.isfinite(cup_depths) & (cup_depths > 0.05)]
        z_m = float(np.median(valid_depths)) if len(valid_depths) > 0 else OBJECT_MAX_DEPTH

        x_m = (cx_px - CX) * z_m / FX
        y_m = (cy_px - CY) * z_m / FY

        pose = PoseStamped()
        pose.header.frame_id    = "sim_camera"
        pose.header.stamp       = self.get_clock().now().to_msg()
        pose.pose.position.x    = x_m
        pose.pose.position.y    = y_m
        pose.pose.position.z    = z_m
        pose.pose.orientation.w = 1.0
        self.pub_pose.publish(pose)

        self.get_logger().info(f"Cup at px=({cx_px:.0f},{cy_px:.0f}) z={z_m:.3f}m")

        debug       = rgb.copy()
        region_mask = ((labels == best_label) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)
        cv2.circle(debug, (int(cx_px), int(cy_px)), 6, (0, 255, 255), -1)
        cv2.putText(debug, f"z={z_m:.3f}m", (int(cx_px) + 8, int(cy_px) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self._publish_debug(debug, mask)

    def _publish_debug(self, rgb, mask):
        try:
            combined = np.hstack([rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(combined, encoding="bgr8"))
        except Exception as e:
            self.get_logger().warn(f"Debug publish failed: {e}")


def main():
    rclpy.init()
    node = CupDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
