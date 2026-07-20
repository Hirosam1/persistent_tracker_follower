import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import supervision as sv

from ai.extractor import ReIDExtractor

# ---------------------------------------------------------------------------
# Target state machine
# ---------------------------------------------------------------------------

class TargetState(Enum):
    INITIALIZING = auto()
    TRACKING = auto()
    LOST = auto()
    SEARCHING = auto()


@dataclass
class _CalibratedTarget:
    N_CALIBRATED_SAMPLES=7
    """Stores a calibrated reference for the person we are following."""
    feature_history: list[np.ndarray] = field(default_factory=list)

    def add_feature(self, feat: np.ndarray) -> bool:
        self.feature_history.append(feat)
        # limit memory to the 3 most recent calibrated features
        if len(self.feature_history) > _CalibratedTarget.N_CALIBRATED_SAMPLES:
            self.feature_history.pop(0)
        return True

    def is_ready(self) -> bool:
        return len(self.feature_history) >= _CalibratedTarget.N_CALIBRATED_SAMPLES


@dataclass
class _ActiveTarget:
    """
    Tracks the current (most recent) appearance and kinematic state
    of the person we are following.
    """
    track_id: int = -1
    state: TargetState = TargetState.INITIALIZING
    feature_history: list[np.ndarray] = field(default_factory=list)
    last_xyxy: tuple[float, float, float, float] | None = None
    last_seen: float = 0.0
    velocity: tuple[float, float] = (0.0, 0.0)
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    t_history: deque = field(default_factory=lambda: deque(maxlen=30))
    search_attempts: int = 0


# ---------------------------------------------------------------------------
# Target Manager
# ---------------------------------------------------------------------------

class TargetManager:
    """
    Maintains a persistent target identity on top of a frame-to-frame tracker.

    When the underlying tracker loses or re-assigns the target ID, this manager
    uses ReID appearance features to re-acquire the correct person.  A separate
    *calibrated* reference is built during the first few seconds to provide a
    high-confidence anchor for re-identification.
    """

    def __init__(
        self,
        reid: ReIDExtractor,
        sim_threshold: float = 0.35,
        calibrated_sim_threshold: float = 0.71,
        feature_history_size: int = 6,
        search_expand_ratio: float = 2.0,
        full_frame_search: bool = False,
        use_calibrated_only: bool = True,
    ) -> None:
        self.reid = reid
        self.sim_threshold = sim_threshold
        self.calibrated_sim_threshold = calibrated_sim_threshold
        self.feature_history_size = feature_history_size
        self.search_expand_ratio = search_expand_ratio
        self.full_frame_search = full_frame_search
        self.use_calibrated_only = use_calibrated_only

        self.target = _ActiveTarget()
        self.calibrated = _CalibratedTarget()
        self.printer = print

    # -- public properties ---------------------------------------------------

    @property
    def track_id(self) -> int:
        return self.target.track_id

    @property
    def state(self) -> TargetState:
        return self.target.state

    # -- public interface -----------------------------------------------------

    def reset(self) -> None:
        self.target = _ActiveTarget()
        self.calibrated = _CalibratedTarget()

    def designate(self, detections: sv.Detections, frame: np.ndarray,
                  det_idx: int = 0) -> bool:
        if detections.tracker_id is None or len(detections) <= det_idx:
            return False
        crop = self._crop(frame, detections.xyxy[det_idx])
        if crop is None or crop.shape[0] < 20 or crop.shape[1] < 20:
            return False

        feat = self.reid.extract(crop)
        self.calibrated = _CalibratedTarget()
        self.target = _ActiveTarget(
            track_id=int(detections.tracker_id[det_idx]),
            state=TargetState.TRACKING,
            feature_history=[feat],
            last_xyxy=tuple(detections.xyxy[det_idx]),
            last_seen=time.time(),
        )
        self.calibrated.add_feature(feat)
        return True

    def update(self, detections: sv.Detections, frame: np.ndarray,
               frame_count: int) -> None:
        now = time.time()
        # -- INITIALISING: auto-select the first detection -------------------
        if self.target.state == TargetState.INITIALIZING:
            if len(detections) > 0:
                self.designate(detections, frame, det_idx=0)
            return

        # -- calibration: collect features for the first N frames ------------
        if not self.calibrated.is_ready() and self.target.state == TargetState.TRACKING:
            if frame_count % 30 == 0:
                if self._calibrate_step(detections, frame):
                    self.printer("[calibration] reference features collected")

        # -- TRACKING / LOST / SEARCHING -------------------------------------
        match = self._find_by_track_id(detections)
        if match is not None:
            self._on_track_found(match, frame, now)
        else:
            self._on_track_lost(detections, frame, now)

    # -- internal: matching --------------------------------------------------

    def _find_by_track_id(self, detections: sv.Detections):
        if detections.tracker_id is None or len(detections) == 0:
            return None
        mask = detections.tracker_id == self.target.track_id
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None
        idx = int(indices[0])
        return (idx, detections.xyxy[idx], int(detections.tracker_id[idx]))

    def _find_by_appearance(self, detections: sv.Detections,
                            frame: np.ndarray) -> tuple | None:
        if len(detections) == 0 or not self.target.feature_history:
            return None

        search_region = (
            None if self.full_frame_search
            else self._predict_search_region()
        )

        # Choose the reference features and threshold.
        # If we have calibrated features, use them with a stricter threshold.
        use_calibrated = self.use_calibrated_only and self.calibrated.is_ready()
        ref_features = (
            self.calibrated.feature_history if use_calibrated
            else self.target.feature_history
        )
        threshold = (
            self.calibrated_sim_threshold if use_calibrated
            else self.sim_threshold
        )

        best_sim = -1.0
        best_idx = -1
        best_xyxy = None
        best_tid = -1

        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            tid = int(detections.tracker_id[i])

            if tid == self.target.track_id:
                continue
            if search_region is not None and not self._inside(xyxy, search_region):
                continue

            crop = self._crop(frame, xyxy)
            if crop is None or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            feat = self.reid.extract(crop)
            sim = float(np.max(
                [self.reid.similarity(feat, ref) for ref in ref_features]
            ))

            if sim > best_sim and tid >= 0:
                best_sim = sim
                best_idx = i
                best_xyxy = xyxy
                best_tid = tid

        if best_sim >= threshold and best_idx >= 0:
            return (best_idx, best_xyxy, best_tid, best_sim)
        return None

    # -- internal: actions ---------------------------------------------------

    def _on_track_found(self, match, frame: np.ndarray, now: float) -> None:
        idx, xyxy, tid = match
        self.target.state = TargetState.TRACKING
        self.target.track_id = tid
        self.target.last_xyxy = tuple(xyxy)
        self.target.last_seen = now
        self.target.bbox_history.append(xyxy)
        self.target.t_history.append(now)
        self._update_velocity()
        self.target.search_attempts = 0

        if len(self.target.bbox_history) % 15 == 0:
            self._append_feature(frame, xyxy)

    def _on_track_lost(self, detections: sv.Detections,
                       frame: np.ndarray, now: float) -> None:
        elapsed = now - self.target.last_seen

        if elapsed < 1.0:
            self.target.state = TargetState.LOST
            return

        if self.target.state != TargetState.SEARCHING:
            self.target.state = TargetState.SEARCHING

        match = self._find_by_appearance(detections, frame)
        if match is not None:
            idx, xyxy, tid, sim = match
            self.target.track_id = tid
            self.target.state = TargetState.TRACKING
            self.target.last_xyxy = tuple(xyxy)
            self.target.last_seen = now
            self.target.bbox_history.append(xyxy)
            self.target.t_history.append(now)
            self._update_velocity()
            self._append_feature(frame, xyxy)
            self.target.search_attempts += 1
            self.printer(f"[ReID] target re-acquired  track={tid}  sim={sim:.3f}")

    def _calibrate_step(self, detections: sv.Detections,
                        frame: np.ndarray) -> bool:
        """Collect one more feature for the calibrated reference."""
        if len(detections) == 0:
            return False
        crop = self._crop(frame, detections.xyxy[0])
        if crop is None or crop.shape[0] < 20 or crop.shape[1] < 20:
            return False
        feat = self.reid.extract(crop)
        self.calibrated.add_feature(feat)
        return True

    # -- helpers -------------------------------------------------------------

    def _append_feature(self, frame: np.ndarray, xyxy) -> None:
        crop = self._crop(frame, xyxy)
        if crop is not None:
            feat = self.reid.extract(crop)
            self.target.feature_history.append(feat)
            if len(self.target.feature_history) > self.feature_history_size:
                self.target.feature_history.pop(0)

    def _update_velocity(self) -> None:
        if len(self.target.bbox_history) < 2:
            return
        p = self.target.bbox_history[-2]
        c = self.target.bbox_history[-1]
        pc = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        cc = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2

        dt = self.target.t_history[-1] - self.target.t_history[-2]
        dt = max(dt, 1e-4)
        self.target.velocity = (
            (cc[0] - pc[0]) / dt,
            (cc[1] - pc[1]) / dt,
        )

    def _predict_search_region(self):
        lp = self.target.last_xyxy
        if lp is None:
            return None
        x1, y1, x2, y2 = lp
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1

        vx, vy = self.target.velocity
        pcx = cx + vx * self.search_expand_ratio
        pcy = cy + vy * self.search_expand_ratio

        ew = max(w * self.search_expand_ratio, 100)
        eh = max(h * self.search_expand_ratio, 100)

        return (pcx - ew / 2, pcy - eh / 2, pcx + ew / 2, pcy + eh / 2)

    @staticmethod
    def _crop(frame: np.ndarray, xyxy) -> np.ndarray | None:
        x1, y1, x2, y2 = map(int, xyxy)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _inside(xyxy, region) -> bool:
        rx1, ry1, rx2, ry2 = region
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2
