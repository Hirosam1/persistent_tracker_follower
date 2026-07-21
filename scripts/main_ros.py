#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

import math
import time

from ultralytics import YOLO
from cv_bridge import CvBridge
import supervision as sv
import numpy as np

from scripts.ai.trackers import build_tracker, NEEDS_FRAME
from scripts.ai.extractor import ReIDExtractor

from scripts.target_manager import TargetManager, TargetState

from scripts.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_TRACKER,
    DEFAULT_VIDEO_SOURCE,
    MODEL_PATH,
    REID_FEATURE_HISTORY_SIZE,
    REID_SEARCH_EXPAND_RATIO,
    REID_SIMILARITY_THRESHOLD,
    REID_CALIBRATED_SIM_THRESHOLD,
    REID_USE_CALIBRATED_ONLY,
)

FRAME_COUNT_LOOP=50000
FRAME_TIME_HISTORY_SIZE=30*5

TRACKER_EXPECTED_FPS=9

class PersistentTrackerNode(Node):
    def __init__(self):
        super().__init__('persistent_tracker')
        self.get_logger().info("Starting tracker follower node...")
        # ── Parameters ───────
        self.declare_parameter('yolo_confidence', DEFAULT_CONFIDENCE)
        self.declare_parameter('tracker', DEFAULT_TRACKER)
        self.declare_parameter('reid_feature_history_size', REID_FEATURE_HISTORY_SIZE)
        self.declare_parameter('reid_calibrated_sim_threshold', REID_CALIBRATED_SIM_THRESHOLD)

        self.yolo_confidence               = self.get_parameter('yolo_confidence').value
        tracker_name                       = self.get_parameter('tracker').value
        reid_feature_history_size     = self.get_parameter('reid_feature_history_size').value
        reid_calibrated_sim_threshold = self.get_parameter('reid_calibrated_sim_threshold').value
        # ── Components ───────
        self.bridge = CvBridge()
        self.get_logger().info(f"Loading yolo model: {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f"Loading tracker: {tracker_name}...")
        self.tracker = build_tracker(tracker_name, TRACKER_EXPECTED_FPS)
        self.needs_frame = tracker_name in NEEDS_FRAME
        self.frame_times = []
        self.last_frame_time = time.perf_counter()
        self.is_detection_enabled = True

        try:
            self.reid = ReIDExtractor()
            self.get_logger().info(f"ReId initiated device={self.reid.device}")
        except Exception as exc:
            self.get_logger().error(f"FAILED: {exc}")
            self.reid = None
        
        self.get_logger().info("Loading target manager...")
        self.target_mgr = (
            TargetManager(
                reid=self.reid,
                sim_threshold=REID_SIMILARITY_THRESHOLD,
                calibrated_sim_threshold=reid_calibrated_sim_threshold,
                feature_history_size=reid_feature_history_size,
                search_expand_ratio=REID_SEARCH_EXPAND_RATIO,
                full_frame_search=True,
                use_calibrated_only=REID_USE_CALIBRATED_ONLY)
            if self.reid is not None else None)
        self.target_mgr.printer = self.get_logger().info
        self.camera_info = None
        self.frame_count = 0

        # ── Communication ───────
        self.create_subscription(Image, 'camera/image', self._image_cb, 10)
        self.create_subscription(CameraInfo, 'camera/camera_info', self._camera_info_cb, 10)
        self.create_subscription(String, 'follower/reset_target', self._reset_target_cb, 10)
        self.create_subscription(Bool, 'follower/set_detection', self._set_detection_cb, 10)

        self.person_pose_pub = self.create_publisher(PoseStamped, 'person_pose', 10)
        self.get_logger().info("Finished starting node.")


    @staticmethod
    def _make_pose_stamped(x: float, y: float, yaw: float, stamp) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id      = 'base_link'
        ps.header.stamp         = stamp
        ps.pose.position.x      = x
        ps.pose.position.y      = y
        ps.pose.position.z      = 0.0
        ps.pose.orientation.z   = math.sin(yaw / 2.0)
        ps.pose.orientation.w   = math.cos(yaw / 2.0)
        return ps

    @staticmethod
    def _calc_fps(frames_times):
        return 1.0/np.mean(frames_times)
    
    def _reset_target_cb(self, msg: String):
        self.get_logger().info("Resetting target...")
        self.target_mgr.reset()

    def _set_detection_cb(self, msg: Bool):
        self.is_detection_enabled = msg.data
        self.get_logger().info(f"Setting person detection to: {self.is_detection_enabled}")


    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = {"width": msg.width, "height": msg.height, "fov": 47}
            self.get_logger().info(f"Got camera info: {self.camera_info}")

    
    def _process_image_msg(self, image_msg: Image):
        # Convert to grey-scale
        try:
            cv_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return
        self.frame_count = (self.frame_count + 1)% FRAME_COUNT_LOOP
        # --- detect person with YOLO ---
        results = next(self.model.predict(
            cv_img, conf=self.yolo_confidence, classes=[0], verbose=False, stream=True))
        detections = sv.Detections.from_ultralytics(results)
        # --- track ---
        detections = self.tracker.update(
            detections=detections,
            frame=cv_img if self.needs_frame else None)
        # --- target manager ---
        if self.target_mgr is not None:
            self.target_mgr.update(detections, cv_img, self.frame_count)
        else:
            return
        
        if self.target_mgr.target.last_xyxy is not None and self.camera_info is not None\
            and self.target_mgr.target.state == TargetState.TRACKING:
            FIXED_DIST=1.0
            IMG_WIDTH=self.camera_info['width']
            CAMERA_FOV_H=np.deg2rad(47.5)/2.0
            x1, y1, x2, y2 = self.target_mgr.target.last_xyxy
            target_x_center_norm = ((x2-x1)/2+x1)/IMG_WIDTH
            target_angle = (2*CAMERA_FOV_H*target_x_center_norm)-(CAMERA_FOV_H)
            x = math.cos(target_angle)*FIXED_DIST
            y = math.sin(target_angle)*FIXED_DIST
            self.get_logger().info(f"Detect target at x:{x:.2f}, y:{y:.2f}, yawn:{np.rad2deg(target_angle):.2f}", 
                                   throttle_duration_sec=2.5)
            msg_out = PersistentTrackerNode._make_pose_stamped(x,y,target_angle,
                                                            self.get_clock().now().to_msg())
        
            self.person_pose_pub.publish(msg_out)


    def _image_cb(self, msg: Image):
        if(self.is_detection_enabled):
            self._process_image_msg(msg)
        
        if(len(self.frame_times) < FRAME_TIME_HISTORY_SIZE):
            self.frame_times.append(time.perf_counter() - self.last_frame_time)
        else:
            self.frame_times.pop(0)
            self.frame_times.append(time.perf_counter() - self.last_frame_time)
        self.last_frame_time = time.perf_counter()
        self.get_logger().info(F"FPS: {PersistentTrackerNode._calc_fps(self.frame_times)}",
                               throttle_duration_sec=5.0)


def main_ros(args=None):
    rclpy.init(args=args)
    node = PersistentTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main_ros()
