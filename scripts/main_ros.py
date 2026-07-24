#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, CameraInfo,  LaserScan
from geometry_msgs.msg import PoseStamped

import math
import time
from collections import deque

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
FRAME_TIME_HISTORY_SIZE=15*5

TRACKER_EXPECTED_FPS=13

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
        self.model = YOLO(MODEL_PATH, task='detect')
        self.get_logger().info(f"Loading tracker: {tracker_name}...")
        self.tracker = build_tracker(tracker_name, TRACKER_EXPECTED_FPS)
        self.needs_frame = tracker_name in NEEDS_FRAME
        self.proc_times = {'frame':       deque(maxlen=FRAME_TIME_HISTORY_SIZE), 
                            'yolo':       deque(maxlen=FRAME_TIME_HISTORY_SIZE),
                            'track':      deque(maxlen=FRAME_TIME_HISTORY_SIZE),
                            'target_mgr': deque(maxlen=FRAME_TIME_HISTORY_SIZE)}
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
        self.latest_scan = None
        self.last_frame_t = time.perf_counter()
        # ── Communication ───────
        self.create_subscription(Image, 'camera/image', self._image_cb, 10)
        self.create_subscription(CameraInfo, 'camera/camera_info', self._camera_info_cb, 10)
        self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        self.create_subscription(String, 'follower/reset_target', self._reset_target_cb, 10)
        self.create_subscription(Bool, 'follower/set_detection', self._set_detection_cb, 10)

        self.person_pose_pub = self.create_publisher(PoseStamped, 'person_pose', 10)
        self._ema_angle = 0.0
        self._ema_alpha = 0.4
        self.get_logger().info("Finished starting node!\n"
                               f"target FPS: {TRACKER_EXPECTED_FPS}\n"
                               f"Cab. ReId threshold: {reid_calibrated_sim_threshold}\n"
                               f"Detection confidence: {self.yolo_confidence}\n"
                               f"ReId feature history size: {reid_feature_history_size}")


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

    # -- callbacks  ----------------------------------------------------------
    def _reset_target_cb(self, msg: String):
        self.get_logger().info("Resetting target...")
        self.target_mgr.reset()


    def _set_detection_cb(self, msg: Bool):
        self.is_detection_enabled = msg.data
        self.get_logger().info(f"Setting person detection to: {self.is_detection_enabled}")


    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = {"width": msg.width, "height": msg.height, "fov": 80}
            self.get_logger().info(f"Got camera info: {self.camera_info}")


    def _image_cb(self, msg: Image):
        self.proc_times['frame'].append(time.perf_counter() - self.last_frame_t)
        self.last_frame_t = time.perf_counter()
        if(self.is_detection_enabled):
            self._process_image_msg(msg)

        self.get_logger().info(f"FPS: {1.0/np.mean(self.proc_times['frame']):.1f}\n\""
                                f"yolo: {np.mean(self.proc_times['yolo']):.2f}\n"
                                f"track: {np.mean(self.proc_times['track']):.2f}\n"
                                f"target_mgr: {np.mean(self.proc_times['target_mgr']):.2f}",
                               throttle_duration_sec=15.0)


    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _get_scan_distance(self, angle: float, window: int = 5, fallback: float = 1.0) -> float:
        if self.latest_scan is None:
            return fallback
        s = self.latest_scan
        span = s.angle_max - s.angle_min
        is_360 = span > 6.2
        if is_360:
            angle = angle % (2.0 * math.pi)
            if angle < s.angle_min:
                angle += 2.0 * math.pi
        if angle < s.angle_min or angle > s.angle_max:
            return fallback
        idx = int((angle - s.angle_min) / s.angle_increment)
        n = len(s.ranges)
        valid = []
        for off in range(-window, window + 1):
            i = (idx + off) % n if is_360 else idx + off
            if not is_360 and (i < 0 or i >= n):
                continue
            r = s.ranges[i]
            if s.range_min < r < s.range_max:
                valid.append(r)
        return min(valid) if valid else fallback

    # -- processing  ---------------------------------------------------------
    def _process_image_msg(self, image_msg: Image):
        # Convert to cv image
        try:
            cv_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return
        self.frame_count = (self.frame_count + 1)% FRAME_COUNT_LOOP
        # --- detect person with YOLO ---
        start_time = time.perf_counter()
        results = next(self.model.predict(
        cv_img, conf=self.yolo_confidence, classes=[0], verbose=False, stream=True))
        self.proc_times['yolo'].append(time.perf_counter() - start_time)
        detections = sv.Detections.from_ultralytics(results)
        # --- track ---
        start_time = time.perf_counter()
        detections = self.tracker.update(
            detections=detections,
            frame=cv_img if self.needs_frame else None)
        self.proc_times['track'].append(time.perf_counter() - start_time)
        # --- target manager ---
        if self.target_mgr is not None:
            start_time = time.perf_counter()
            self.target_mgr.update(detections, cv_img, self.frame_count)
            self.proc_times['target_mgr'].append(time.perf_counter() - start_time)
        else:
            return

        if len(self.target_mgr.target.bbox_history) >= 3 and self.camera_info is not None\
            and self.target_mgr.target.state == TargetState.TRACKING:
            IMG_WIDTH=self.camera_info['width']
            CAMERA_FOV_H=np.deg2rad(self.camera_info['fov'])/2.0
            CUT_OUT_THRES=0.1
            DIST_REDUCTION=0.9
            #x1, y1, x2, y2 = self.target_mgr.target.last_xyxy
            x1, y1, x2, y2 = TargetManager._average_bboxes(
                                                self.target_mgr.target.bbox_history)
            target_x_center_norm = ((x2-x1)/2+x1)/IMG_WIDTH
            if(target_x_center_norm > CUT_OUT_THRES and target_x_center_norm < 1.0-CUT_OUT_THRES):
                target_angle = -((2*CAMERA_FOV_H*target_x_center_norm)-(CAMERA_FOV_H))
                self._ema_angle = self._ema_alpha * target_angle + (1.0 - self._ema_alpha) * self._ema_angle
                scan_dist = self._get_scan_distance(self._ema_angle)
                x = math.cos(self._ema_angle) *scan_dist*DIST_REDUCTION
                y = math.sin(self._ema_angle) *scan_dist*DIST_REDUCTION
                self.get_logger().info(f"Detect target at x: {x:.2f}, y: {y:.2f}, yawn: {np.rad2deg(self._ema_angle):.2f}", 
                                    throttle_duration_sec=5.0)
                msg_out = PersistentTrackerNode._make_pose_stamped(x,y,self._ema_angle,
                                                                self.get_clock().now().to_msg())
                self.person_pose_pub.publish(msg_out)


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
