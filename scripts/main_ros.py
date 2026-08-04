#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import String, Bool, Empty, Float32
from sensor_msgs.msg import Image, CameraInfo,  LaserScan
from geometry_msgs.msg import PoseStamped

import math
import time
from collections import deque
from pathlib import Path
import shutil

from ultralytics import YOLO
from cv_bridge import CvBridge
import cv2
import supervision as sv
import numpy as np

from scripts.ai.trackers import build_tracker, NEEDS_FRAME
from scripts.ai.extractor import ReIDExtractor, ReIDExtractorOpenVINO
from scripts.annotator import SceneAnnotator

from scripts.target_manager import TargetManager, TargetState

from scripts.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_TRACKER,
    MODEL_PATH,
    REID_FEATURE_HISTORY_SIZE,
    REID_SEARCH_EXPAND_RATIO,
    REID_SIMILARITY_THRESHOLD,
    REID_CALIBRATED_SIM_THRESHOLD,
    REID_USE_CALIBRATED_ONLY,
    REID_VERIFICATION_INTERVAL,
    CONFIDENCE_BOOST_CALIBRATED,
    CONFIDENCE_BOOST_HISTORY,
    CONFIDENCE_LOST_DECAY_RATE,
    CONFIDENCE_REACQUIRE_PENALTY,
    OVERLAP_IOU_THRESHOLD,
    OVERLAP_PENALTY,
    EVIDENCE_NORMAL_VOTES,
    EVIDENCE_SUSPICIOUS_VOTES,
    EVIDENCE_LOCKED_VOTES,
    SUSPICION_EVIDENCE_THRESHOLD,
    SUSPICION_VERIFY_THRESHOLD,
    SUSPICION_FREEZE_THRESHOLD,
    SUSPICION_DECAY_RATE,
    SUSPICION_AMBIGUITY_GAP,
    SUSPICION_OVERLAP_WEIGHT,
    SUSPICION_JUMP_WEIGHT,
    SUSPICION_AREA_WEIGHT,
    SUSPICION_ID_MISS_WEIGHT,
    SUSPICION_NEW_PERSON_WEIGHT,
    SUSPICION_CONF_DROP_WEIGHT,
    SUSPICION_AMBIGUITY_WEIGHT,
    SUSPICION_DRIFT_WEIGHT,
    BBOX_JUMP_PIXELS,
    BBOX_AREA_CHANGE_RATIO,
    NEW_PERSON_PROXIMITY_PX,
    CONF_DROP_THRESHOLD,
    LOCK_CONFIDENCE,
    UNLOCK_CONFIDENCE,
    LOCK_MIN_STABLE_FRAMES,
    LOCK_LOST_GRACE,
    LEARNING_FREEZE_SECONDS,
    REACQ_LEARNING_COOLDOWN,
    EVENT_VERIFY_MIN_INTERVAL,
    CONFIDENCE_SUSPICION_PENALTY,
    APPEARANCE_CAL_WEIGHT,
    EMBED_CACHE_TTL,
    CREATE_DEBUG_IMGS,
    DEBUG_IMGS_FPS,
    DEBUG_RESIZE_FACTOR,
    DEBUG_FOLDER
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
        self.tracker_name: str             = self.get_parameter('tracker').value
        reid_feature_history_size     = self.get_parameter('reid_feature_history_size').value
        reid_calibrated_sim_threshold = self.get_parameter('reid_calibrated_sim_threshold').value

        out_path = Path(DEBUG_FOLDER)        
        if out_path.exists():
            shutil.rmtree(out_path)  # Deletes folder and all contents
        out_path.mkdir(parents=True, exist_ok=True)
        # ── Components ───────
        self.bridge = CvBridge()
        self.get_logger().info(f"Loading yolo model: {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH, task='detect')
        self.get_logger().info(f"Loading tracker: {self.tracker_name}...")
        self.tracker = build_tracker(self.tracker_name, TRACKER_EXPECTED_FPS)
        self.needs_frame = self.tracker_name in NEEDS_FRAME
        self.proc_times = {'frame':       deque(maxlen=FRAME_TIME_HISTORY_SIZE), 
                            'yolo':       deque(maxlen=FRAME_TIME_HISTORY_SIZE),
                            'track':      deque(maxlen=FRAME_TIME_HISTORY_SIZE),
                            'target_mgr': deque(maxlen=FRAME_TIME_HISTORY_SIZE)}
        try:
            #self.reid = ReIDExtractor()
            self.reid = ReIDExtractorOpenVINO()
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
                use_calibrated_only=REID_USE_CALIBRATED_ONLY,
                verification_interval=REID_VERIFICATION_INTERVAL,
                confidence_boost_calibrated=CONFIDENCE_BOOST_CALIBRATED,
                confidence_boost_history=CONFIDENCE_BOOST_HISTORY,
                confidence_lost_decay_rate=CONFIDENCE_LOST_DECAY_RATE,
                confidence_reacquire_penalty=CONFIDENCE_REACQUIRE_PENALTY,
                overlap_iou_threshold=OVERLAP_IOU_THRESHOLD,
                overlap_penalty=OVERLAP_PENALTY,
                evidence_normal_votes=EVIDENCE_NORMAL_VOTES,
                evidence_suspicious_votes=EVIDENCE_SUSPICIOUS_VOTES,
                evidence_locked_votes=EVIDENCE_LOCKED_VOTES,
                suspicion_evidence_threshold=SUSPICION_EVIDENCE_THRESHOLD,
                suspicion_verify_threshold=SUSPICION_VERIFY_THRESHOLD,
                suspicion_freeze_threshold=SUSPICION_FREEZE_THRESHOLD,
                suspicion_decay_rate=SUSPICION_DECAY_RATE,
                suspicion_ambiguity_gap=SUSPICION_AMBIGUITY_GAP,
                suspicion_overlap_weight=SUSPICION_OVERLAP_WEIGHT,
                suspicion_jump_weight=SUSPICION_JUMP_WEIGHT,
                suspicion_area_weight=SUSPICION_AREA_WEIGHT,
                suspicion_id_miss_weight=SUSPICION_ID_MISS_WEIGHT,
                suspicion_new_person_weight=SUSPICION_NEW_PERSON_WEIGHT,
                suspicion_conf_drop_weight=SUSPICION_CONF_DROP_WEIGHT,
                suspicion_ambiguity_weight=SUSPICION_AMBIGUITY_WEIGHT,
                suspicion_drift_weight=SUSPICION_DRIFT_WEIGHT,
                bbox_jump_pixels=BBOX_JUMP_PIXELS,
                bbox_area_change_ratio=BBOX_AREA_CHANGE_RATIO,
                new_person_proximity_px=NEW_PERSON_PROXIMITY_PX,
                conf_drop_threshold=CONF_DROP_THRESHOLD,
                lock_confidence=LOCK_CONFIDENCE,
                unlock_confidence=UNLOCK_CONFIDENCE,
                lock_min_stable_frames=LOCK_MIN_STABLE_FRAMES,
                lock_lost_grace=LOCK_LOST_GRACE,
                learning_freeze_seconds=LEARNING_FREEZE_SECONDS,
                reacq_learning_cooldown=REACQ_LEARNING_COOLDOWN,
                event_verify_min_interval=EVENT_VERIFY_MIN_INTERVAL,
                confidence_suspicion_penalty=CONFIDENCE_SUSPICION_PENALTY,
                appearance_cal_weight=APPEARANCE_CAL_WEIGHT,
                embed_cache_ttl=EMBED_CACHE_TTL)
            if self.reid is not None else None)
        self.target_mgr.printer = self.get_logger().info
        self.camera_info = None
        self.camera_info_msg = None
        self.frame_count = 0
        self.latest_scan = None
        self.last_frame_t = time.perf_counter()
        self.calib_request: bool = False
        self.is_detection_enabled = False
        self.last_debug_img = time.perf_counter() if CREATE_DEBUG_IMGS else None
        self.annotator = SceneAnnotator() if CREATE_DEBUG_IMGS else None
        # ── Communication ───────
        self.image_sub = None
        self.scan_sub = None
        self.create_subscription(CameraInfo, 'camera/camera_info', self._camera_info_cb, 10)
        self.create_subscription(Empty, 'tracker/reset_target', self._reset_target_cb, 10)
        self.create_subscription(Bool, 'tracker/set_detection', self._set_detection_cb, 10)

        self.target_ready_pub = self.create_publisher(Empty, 'tracker/target_ready', 10)
        self.person_pose_pub = self.create_publisher(PoseStamped, 'person_pose', 10)
        self.confidence_pub = self.create_publisher(Float32, 'tracker/confidence', 10)
        self.suspicion_pub = self.create_publisher(Float32, 'tracker/suspicion', 10)
        self.identity_lock_pub = self.create_publisher(Bool, 'tracker/identity_lock', 10)
        self.tracker_debug_img_pub = self.create_publisher(Image, 'tracker/debug_img', 10)
        self._ema_angle = 0.0
        self._ema_alpha = 0.32
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
    def _reset_target_cb(self, msg: Empty):
        self.get_logger().info("Resetting target...")
        self.calib_request = True
        self.target_mgr.reset()


    def _set_detection_cb(self, msg: Bool):
        self.is_detection_enabled = msg.data
        if(self.is_detection_enabled):
            self.get_logger().info(f"Enabling person detection")
            self.image_sub = self.create_subscription(Image, 'camera/image', self._image_cb, 10)
            self.scan_sub = self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        else:
            self.get_logger().info(f"Disabling person detection")
            if(self.image_sub is not None):
                self.destroy_subscription(self.image_sub)
                self.image_sub = None
            if(self.scan_sub is not None):
                self.destroy_subscription(self.scan_sub)
                self.scan_sub = None

    def _camera_info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = {"width": msg.width, "height": msg.height, "fov": 65}
            self.camera_info_msg = msg
            self.get_logger().info(f"Got camera info: {self.camera_info}")


    def _image_cb(self, msg: Image):
        self.proc_times['frame'].append(time.perf_counter() - self.last_frame_t)
        self.last_frame_t = time.perf_counter()
        if(self.is_detection_enabled):
            cv_img = None
            # Convert to cv image
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv_img = self.rectify_image(cv_img)
            except Exception as e:
                self.get_logger().warn(f'cv_bridge error: {e}')
                return
            detections = self._process_image_msg(cv_img)
            if(self.calib_request and self.target_mgr.calibrated.is_ready()):
                self.calib_request = False
                self.target_ready_pub.publish(Empty())
                self.get_logger().info("Target calibrated!")
            
            if(CREATE_DEBUG_IMGS and self.last_frame_t - self.last_debug_img > 1.0/DEBUG_IMGS_FPS):
                self.last_debug_img = self.last_frame_t
                annotated = self.annotator.annotate(cv_img, detections, self.model.names, self.target_mgr)
                fps = 1.0/np.mean(self.proc_times['frame'])
                annotated = self.annotator.draw_hud(annotated, self.tracker_name, self.target_mgr, fps)
                annotated = cv2.resize(annotated, dsize=(0,0), fx=DEBUG_RESIZE_FACTOR, fy=DEBUG_RESIZE_FACTOR)
                cv2.imwrite(f"{DEBUG_FOLDER}/debug_img{self.frame_count}.png", annotated)
                #out_img = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                #self.tracker_debug_img_pub.publish(out_img)

        self.get_logger().info(f"FPS: {1.0/np.mean(self.proc_times['frame']):.1f}\n"
                                f"yolo: {np.mean(self.proc_times['yolo']):.2f}\n"
                                f"track: {np.mean(self.proc_times['track']):.2f}\n"
                                f"target_mgr: {np.mean(self.proc_times['target_mgr']):.2f}",
                               throttle_duration_sec=15.0)


    def _scan_cb(self, msg: LaserScan):
        if (self.is_detection_enabled):
            n = len(msg.ranges)
            back_idx = int((math.pi - msg.angle_min) / msg.angle_increment) % n
            self._scan_ranges = np.roll(np.array(msg.ranges, dtype=np.float32), -back_idx)
            self._scan_angle_min = msg.angle_min + back_idx * msg.angle_increment
            self._scan_angle_inc = msg.angle_increment
            self._scan_range_min = msg.range_min
            self._scan_range_max = msg.range_max


    def _get_scan_distance(self, angle: float, window: int = 6, fallback: float = 1.1) -> float:
        if not hasattr(self, '_scan_ranges'):
            return fallback
        angle_norm = angle % (2.0 * math.pi)
        if angle_norm < self._scan_angle_min:
            angle_norm += 2.0 * math.pi
        idx = int((angle_norm - self._scan_angle_min) / self._scan_angle_inc)
        n = len(self._scan_ranges)
        lo = max(0, idx - window)
        hi = min(n, idx + window + 1)
        valid = [r for r in self._scan_ranges[lo:hi] if self._scan_range_min < r < self._scan_range_max]
        return float(min(valid)) if valid else fallback

    # -- processing  ---------------------------------------------------------
    def _process_image_msg(self, cv_img: np.ndarray):
        self.frame_count = (self.frame_count + 1)% FRAME_COUNT_LOOP
        # --- detect person with YOLO ---
        start_time = time.perf_counter()
        results = next(self.model.predict(
        cv_img, conf = self.yolo_confidence, classes=[0], verbose=False, stream=True))
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
            return None

        self.confidence_pub.publish(Float32(data=self.target_mgr.confidence))
        self.suspicion_pub.publish(Float32(data=self.target_mgr.suspicion))
        self.identity_lock_pub.publish(Bool(data=self.target_mgr.identity_locked))

        if (len(self.target_mgr.target.bbox_history) >= 3
            and self.camera_info is not None
            and self.target_mgr.target.state == TargetState.TRACKING
            and self.target_mgr.confidence >= 0.3):
            IMG_WIDTH=self.camera_info['width']
            CAMERA_FOV_H=np.deg2rad(self.camera_info['fov'])/2.0
            CUT_OUT_THRES=0.1
            DIST_REDUCTION=0.9
            MAX_DIST=1.5
            #x1, y1, x2, y2 = self.target_mgr.target.last_xyxy
            x1, y1, x2, y2 = TargetManager._average_bboxes(
                                                self.target_mgr.target.bbox_history)
            target_x_center_norm = ((x2-x1)/2+x1)/IMG_WIDTH
            if(target_x_center_norm > CUT_OUT_THRES and target_x_center_norm < 1.0-CUT_OUT_THRES):
                target_angle = -((2*CAMERA_FOV_H*target_x_center_norm)-(CAMERA_FOV_H))
                self._ema_angle = self._ema_alpha * target_angle + (1.0 - self._ema_alpha) * self._ema_angle
                scan_dist = min(self._get_scan_distance(self._ema_angle), MAX_DIST)
                x = math.cos(self._ema_angle) * scan_dist*DIST_REDUCTION
                y = math.sin(self._ema_angle) * scan_dist*DIST_REDUCTION
                self.get_logger().info(f"Detect target at x: {x:.2f}, y: {y:.2f}, yawn: {np.rad2deg(self._ema_angle):.2f}"
                                    f"  confidence: {self.target_mgr.confidence:.2f}",
                                    throttle_duration_sec=5.0)
                msg_out = PersistentTrackerNode._make_pose_stamped(x,y,self._ema_angle,
                                                                self.get_clock().now().to_msg())
                self.person_pose_pub.publish(msg_out)
        return detections

    def rectify_image(self, image: np.ndarray) -> np.ndarray:
        camera_info = self.camera_info_msg
        # Camera matrix
        K = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)
        # Distortion coefficients
        D = np.array(camera_info.d, dtype=np.float64)
        # Rectification matrix
        R = np.array(camera_info.r, dtype=np.float64).reshape(3, 3)
        # Projection matrix
        P = np.array(camera_info.p, dtype=np.float64).reshape(3, 4)
        height, width = image.shape[:2]
        # Compute rectification maps
        map1, map2 = cv2.initUndistortRectifyMap(
            K,
            D,
            R,
            P[:, :3],        # new camera matrix
            (width, height),
            cv2.CV_32FC1
        )
        rectified = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR
        )
        return rectified

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
