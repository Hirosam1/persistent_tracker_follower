import math
import time
from collections import deque, namedtuple
from dataclasses import dataclass, field, fields, replace
from enum import Enum, auto

import numpy as np
import supervision as sv

from scripts.ai.extractor import ReIDExtractor
from scripts.config import TargetManagerConfig

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
    """Stores a calibrated reference for the person we are following.
    Once ready, this set is never modified again."""
    feature_history: list[np.ndarray] = field(default_factory=list)

    def add_feature(self, feat: np.ndarray) -> bool:
        self.feature_history.append(feat)
        if len(self.feature_history) > _CalibratedTarget.N_CALIBRATED_SAMPLES:
            self.feature_history.pop(0)
        return True

    def is_ready(self) -> bool:
        return len(self.feature_history) >= _CalibratedTarget.N_CALIBRATED_SAMPLES


@dataclass
class _ActiveTarget:
    """
    Tracks the current appearance, kinematic state and identity bookkeeping
    of the person we are following.
    """
    track_id: int = -1
    state: TargetState = TargetState.INITIALIZING
    feature_history: list[np.ndarray] = field(default_factory=list)
    last_xyxy: tuple[float, float, float, float] | None = None
    last_seen: float = 0.0
    velocity: tuple[float, float] = (0.0, 0.0)
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=5))
    t_history: deque = field(default_factory=lambda: deque(maxlen=3))

    # -- confidence system ------------------------------------------------
    confidence: float = 0.0
    last_verification_time: float = 0.0
    last_verif_sim_history: float = 0.0
    last_verif_sim_calibrated: float = 0.0

    # -- suspicion / lock / evidence --------------------------------------
    suspicion: float = 0.0
    identity_locked: bool = False
    lock_stable_frames: int = 0
    pending_switch_tid: int = -1
    pending_switch_votes: int = 0
    pending_switch_required: int = -1
    learning_frozen_until: float = 0.0
    consecutive_verifications: int = 0
    last_reacquisition_time: float = 0.0


# A scored detection hypothesis.
Candidate = namedtuple(
    "Candidate", ["idx", "tid", "xyxy", "sim_hist", "sim_cal", "score", "conf"])


# ---------------------------------------------------------------------------
# Target Manager
# ---------------------------------------------------------------------------

class TargetManager:
    """
    Maintains a persistent target identity on top of a frame-to-frame tracker.

    ByteTrack only *proposes* hypotheses (bounding boxes with IDs).  Every
    frame this manager scores each visible candidate by ReID appearance against
    both the calibrated reference and the adaptive history.  Appearance is the
    final authority on identity; track IDs are only a tie-break hint.

    A suspicion score, identity lock and evidence-tier system protect the
    identity from ByteTrack ID swaps and appearance contamination.
    """

    def __init__(
        self,
        reid: ReIDExtractor,
        config: TargetManagerConfig | None = None,
        **overrides,
    ) -> None:
        """ReID scoring and identity-protection manager.

        `config` supplies the tunables (defaults from `TargetManagerConfig`);
        extra keyword `overrides` are applied on top. For backwards
        compatibility every setting is also exposed as an instance attribute.
        """
        cfg = config if config is not None else TargetManagerConfig()
        unknown = set(overrides) - {f.name for f in fields(cfg)}
        if unknown:
            raise TypeError(
                f"Unknown TargetManager option(s): {sorted(unknown)}")
        self.config = replace(cfg, **overrides) if overrides else cfg
        self.reid = reid

        # expose every tuning knob as `self.<name>` for existing call sites
        self.__dict__.update(vars(self.config))

        self.target = _ActiveTarget()
        self.calibrated = _CalibratedTarget()
        self.printer = print

        # cache of (timestamp, embedding) keyed by ByteTrack ID
        self._embed_cache: dict[int, tuple[float, np.ndarray]] = {}
        self._last_update_time: float | None = None

    # -- public properties ---------------------------------------------------

    @property
    def track_id(self) -> int:
        return self.target.track_id

    @property
    def state(self) -> TargetState:
        return self.target.state

    @property
    def confidence(self) -> float:
        return self.target.confidence

    @property
    def suspicion(self) -> float:
        return self.target.suspicion

    @property
    def identity_locked(self) -> bool:
        return self.target.identity_locked

    # -- public interface -----------------------------------------------------

    def reset(self) -> None:
        self.target = _ActiveTarget()
        self.calibrated = _CalibratedTarget()
        self._embed_cache.clear()
        self._last_update_time = None

    def designate(self, detections: sv.Detections, frame: np.ndarray,
                  det_idx: int = 0) -> bool:
        if detections.tracker_id is None or len(detections) <= det_idx:
            return False
        feat = self._extract_feature(frame, detections.xyxy[det_idx])
        if feat is None:
            return False
        self.calibrated = _CalibratedTarget()
        now = time.time()
        self.target = _ActiveTarget(
            track_id=int(detections.tracker_id[det_idx]),
            state=TargetState.TRACKING,
            feature_history=[feat],
            last_xyxy=tuple(detections.xyxy[det_idx]),
            last_seen=now,
            confidence=0.3,
            last_verification_time=now,
        )
        self.calibrated.add_feature(feat)
        self._embed_cache.clear()
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
            if frame_count % 16 == 0:
                if self._calibrate_step(detections, frame):
                    self.printer("[calibration] reference features collected")
                    if self.calibrated.is_ready():
                        self.target.last_verification_time = now
                        self.target.confidence = max(
                            self.target.confidence, 0.5)

        # -- appearance-first candidate scoring ------------------------------
        scored = self._score_candidates(detections, frame)

        # -- suspicion from events -------------------------------------------
        self.target.suspicion = self._compute_suspicion(
            detections, frame, scored, now)

        best = scored[0] if scored else None
        if best is None:
            # no candidate passes the appearance check -> not our person
            self._on_track_lost(detections, frame, now)
            return

        # -- follow the appearance-chosen box --------------------------------
        self._follow(best, frame, now)
        self._maybe_reassociate(best, now)
        self._update_verification(frame, now)
        self._update_lock(now)

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

    def _score_candidates(self, detections: sv.Detections,
                          frame: np.ndarray) -> list[Candidate]:
        """Score every visible candidate against calibrated + history features.
        Candidates failing the appearance check are discarded.
        Returns a list sorted by descending score."""
        out: list[Candidate] = []
        if detections.tracker_id is None or len(detections) == 0:
            return out
        if not self.target.feature_history:
            return out

        has_calibrated = self.calibrated.is_ready()
        search_region = (
            None if self.full_frame_search else self._predict_search_region())

        for i in range(len(detections)):
            tid = int(detections.tracker_id[i])
            xyxy = detections.xyxy[i]
            if search_region is not None and not self._inside(xyxy, search_region):
                continue

            feat = self._embedding_for(tid, frame, xyxy)
            if feat is None:
                continue

            sim_hist = self._max_similarity(feat, self.target.feature_history)
            sim_cal = -1.0
            if has_calibrated and self.calibrated.feature_history:
                sim_cal = self._max_similarity(
                    feat, self.calibrated.feature_history)

            hist_ok = sim_hist >= self.sim_threshold
            if has_calibrated:
                # appearance authority: must satisfy the strict calibrated
                # anchor AND the history consistency check
                cal_ok = sim_cal >= self.calibrated_sim_threshold
                if not (cal_ok and hist_ok):
                    continue
                score = (self.appearance_cal_weight * sim_cal
                         + (1.0 - self.appearance_cal_weight) * sim_hist)
            else:
                if not hist_ok:
                    continue
                score = sim_hist

            # track ID is only a hint: tiny tie-break boost for continuity
            if tid == self.target.track_id:
                score += 0.02

            conf = 1.0
            if detections.confidence is not None and i < len(detections.confidence):
                conf = float(detections.confidence[i])

            out.append(Candidate(idx=i, tid=tid, xyxy=xyxy,
                                 sim_hist=sim_hist, sim_cal=sim_cal,
                                 score=score, conf=conf))

        out.sort(key=lambda c: -c.score)
        return out

    def _find_by_appearance(self, detections: sv.Detections,
                            frame: np.ndarray) -> tuple | None:
        scored = self._score_candidates(detections, frame)
        if not scored:
            return None
        idx, tid, xyxy, sim_hist, sim_cal, _, _ = scored[0]
        matched_calibrated = (self.calibrated.is_ready()
                              and sim_cal >= self.calibrated_sim_threshold)
        return (idx, xyxy, tid, sim_hist, sim_cal, matched_calibrated)

    # -- internal: suspicion -------------------------------------------------

    def _compute_suspicion(self, detections: sv.Detections, frame: np.ndarray,
                           scored: list[Candidate], now: float) -> float:
        """Decay the stored suspicion and add weights for this frame's events."""
        dt = 0.0
        if self._last_update_time is not None:
            dt = max(0.0, now - self._last_update_time)
        self._last_update_time = now

        s = self.target.suspicion * max(
            0.0, 1.0 - self.suspicion_decay_rate * dt)

        best = scored[0] if scored else None
        follow_xyxy = best.xyxy if best else self.target.last_xyxy
        follow_tid = best.tid if best else self.target.track_id

        # 1) bbox overlap with another tracked person
        if self._check_overlap(detections, follow_xyxy, follow_tid):
            s += self.suspicion_overlap_weight

        # 2) abrupt bbox jump + 3) abrupt area change
        if best is not None and self.target.last_xyxy is not None:
            cx, cy = self._center(best.xyxy)
            lcx, lcy = self._center(self.target.last_xyxy)
            if math.hypot(cx - lcx, cy - lcy) > self.bbox_jump_pixels:
                s += self.suspicion_jump_weight
            if self.target.bbox_history:
                avg_area = sum(self._area(b) for b in self.target.bbox_history)
                avg_area /= len(self.target.bbox_history)
                area = self._area(best.xyxy)
                if (avg_area > 0
                        and (area > avg_area * self.bbox_area_change_ratio
                             or area < avg_area / self.bbox_area_change_ratio)):
                    s += self.suspicion_area_weight

        # 4) stored track ID disappeared this frame
        if self._find_by_track_id(detections) is None:
            s += self.suspicion_id_miss_weight

        # 5) a new person appeared next to the target
        if best is not None and self._new_person_nearby(detections, best):
            s += self.suspicion_new_person_weight

        # 6) detection confidence dropped sharply
        if best is not None and best.conf < self.conf_drop_threshold:
            s += self.suspicion_conf_drop_weight

        # 7) appearance ambiguity: two top candidates nearly tied
        if (len(scored) >= 2
                and scored[0].score - scored[1].score < self.suspicion_ambiguity_gap):
            s += self.suspicion_ambiguity_weight

        # 8) appearance drift: last verification matched history but not anchor
        if (self.target.last_verif_sim_calibrated > 0.0
                and self.target.last_verif_sim_calibrated < self.calibrated_sim_threshold
                and self.target.last_verif_sim_history >= self.sim_threshold):
            s += self.suspicion_drift_weight

        return min(1.0, s)

    def _new_person_nearby(self, detections: sv.Detections,
                           best: Candidate) -> bool:
        if detections.tracker_id is None or len(detections) < 2:
            return False
        bx, by = self._center(best.xyxy)
        for i in range(len(detections)):
            if int(detections.tracker_id[i]) == best.tid:
                continue
            cx, cy = self._center(detections.xyxy[i])
            if math.hypot(cx - bx, cy - by) < self.new_person_proximity_px:
                return True
        return False

    # -- internal: identity lock / evidence tiers ----------------------------

    def _required_confirmations(self) -> int:
        """Consecutive confirmations needed to adopt a new identity."""
        if self.target.identity_locked:
            return self.evidence_locked_votes
        if self.target.suspicion >= self.suspicion_evidence_threshold:
            return self.evidence_suspicious_votes
        return self.evidence_normal_votes

    def _update_lock(self, now: float) -> None:
        t = self.target
        stable = t.pending_switch_tid == -1

        if t.identity_locked:
            if (t.confidence < self.unlock_confidence
                    or now - t.last_seen > self.lock_lost_grace):
                t.identity_locked = False
                t.lock_stable_frames = 0
                self.printer("[identity] lock released")
        else:
            if stable and t.suspicion < self.suspicion_evidence_threshold:
                t.lock_stable_frames += 1
            else:
                t.lock_stable_frames = 0
            if (t.confidence >= self.lock_confidence
                    and t.lock_stable_frames >= self.lock_min_stable_frames):
                t.identity_locked = True
                self.printer("[identity] LOCKED")

    # -- internal: actions ---------------------------------------------------

    def _follow(self, best: Candidate, frame: np.ndarray, now: float) -> None:
        t = self.target
        t.state = TargetState.TRACKING
        t.last_xyxy = tuple(best.xyxy)
        t.last_seen = now
        t.bbox_history.append(best.xyxy)
        t.t_history.append(now)
        self._update_velocity()

    def _maybe_reassociate(self, best: Candidate, now: float) -> None:
        t = self.target
        tid = best.tid
        if tid == t.track_id:
            t.pending_switch_tid = -1
            t.pending_switch_votes = 0
            t.pending_switch_required = -1
            return

        # ByteTrack now reports a different ID for our person (possible swap).
        # Do NOT switch immediately: require consecutive confirmations.
        if t.pending_switch_tid == tid:
            t.pending_switch_votes += 1
        else:
            # First frame of the confirmation window: freeze the evidence tier
            # (suspicion / lock state) for the whole window.
            t.pending_switch_tid = tid
            t.pending_switch_votes = 1
            t.pending_switch_required = self._required_confirmations()

        needed = t.pending_switch_required
        if t.pending_switch_votes >= needed:
            self.printer(
                f"[identity] re-associated track={tid} "
                f"after {t.pending_switch_votes} confirmations "
                f"(susp={t.suspicion:.2f}, locked={t.identity_locked})")
            t.track_id = tid
            t.pending_switch_tid = -1
            t.pending_switch_votes = 0
            t.pending_switch_required = -1

    def _update_verification(self, frame: np.ndarray, now: float) -> None:
        t = self.target
        periodic_due = (now - t.last_verification_time
                        >= self.verification_interval)
        event_due = t.suspicion >= self.suspicion_verify_threshold
        if event_due and (now - t.last_verification_time
                          < self.event_verify_min_interval):
            event_due = False  # rate-limit event-driven ReID
        if not (periodic_due or event_due):
            return

        verified, sim_hist, sim_cal, feat = self._verify_current_target(
            frame, t.last_xyxy)
        t.last_verification_time = now
        t.last_verif_sim_history = sim_hist
        t.last_verif_sim_calibrated = sim_cal

        if verified:
            t.consecutive_verifications += 1
            boost = (self.confidence_boost_calibrated
                     if sim_cal >= self.calibrated_sim_threshold
                     else self.confidence_boost_history)
            t.confidence = min(1.0, t.confidence + boost)
            if self._can_learn():
                self._append_feature(frame, t.last_xyxy, feat=feat)
        else:
            t.consecutive_verifications = 0
            self._freeze_learning(now)
            if event_due:
                t.confidence = max(0.0, t.confidence
                                   - self.confidence_suspicion_penalty)

    def _on_track_lost(self, detections: sv.Detections,
                       frame: np.ndarray, now: float) -> None:
        elapsed = now - self.target.last_seen

        # confidence decays while lost
        self.target.confidence = max(
            0.0, self.target.confidence
            - elapsed * self.confidence_lost_decay_rate)
        self._freeze_learning(now)
        self.target.pending_switch_tid = -1
        self.target.pending_switch_votes = 0
        self.target.pending_switch_required = -1

        if elapsed < 1.0:
            self.target.state = TargetState.LOST
            return

        if self.target.state != TargetState.SEARCHING:
            self.target.state = TargetState.SEARCHING

        match = self._find_by_appearance(detections, frame)
        if match is not None:
            idx, xyxy, tid, sim_hist, sim_cal, matched_calibrated = match
            self.target.track_id = tid
            self.target.state = TargetState.TRACKING
            self.target.last_xyxy = tuple(xyxy)
            self.target.last_seen = now
            self.target.bbox_history.append(xyxy)
            self.target.last_reacquisition_time = now

            if matched_calibrated:
                self.target.confidence = max(
                    self.target.confidence, 0.5)
                self.target.last_verification_time = now
                self.target.last_verif_sim_calibrated = sim_cal
                self.target.last_verif_sim_history = sim_hist
            else:
                self.target.confidence = min(
                    self.target.confidence, 0.3)
            self.printer(
                f"[ReID] target re-acquired  track={tid}  "
                f"sim_hist={sim_hist:.3f}  sim_cal={sim_cal:.3f}  "
                f"via_calibrated={matched_calibrated}")

    def _calibrate_step(self, detections: sv.Detections,
                        frame: np.ndarray) -> bool:
        """Collect one more feature for the calibrated reference."""
        if len(detections) == 0:
            return False
        feat = self._extract_feature(frame, detections.xyxy[0])
        if feat is None:
            return False
        self.calibrated.add_feature(feat)
        return True

    # -- learning policy ------------------------------------------------------

    def _can_learn(self) -> bool:
        """Whether the adaptive history may be appended right now."""
        t = self.target
        now = time.time()
        if now < t.learning_frozen_until:
            return False
        if t.suspicion >= self.suspicion_freeze_threshold:
            return False
        if t.pending_switch_tid != -1:
            return False
        if t.state != TargetState.TRACKING:
            return False
        if now - t.last_reacquisition_time < self.reacq_learning_cooldown:
            return False
        if t.consecutive_verifications < self._required_confirmations():
            return False
        return True

    def _freeze_learning(self, now: float) -> None:
        t = self.target
        t.learning_frozen_until = max(
            t.learning_frozen_until, now + self.learning_freeze_seconds)
        t.consecutive_verifications = 0

    # -- verification & overlap ----------------------------------------------

    def _verify_current_target(self, frame: np.ndarray,
                                xyxy) -> tuple[bool, float, float, np.ndarray | None]:
        """Extract a fresh ReID feature from the target crop and check it
        against both the calibrated reference and the history features.
        Returns (verified, sim_to_history, sim_to_calibrated, feat)."""
        if xyxy is None:
            return False, 0.0, 0.0, None
        feat = self._extract_feature(frame, xyxy)
        if feat is None:
            return False, 0.0, 0.0, None

        sim_cal = 0.0
        if self.calibrated.is_ready() and self.calibrated.feature_history:
            sim_cal = self._max_similarity(
                feat, self.calibrated.feature_history)
        sim_hist = 0.0
        if self.target.feature_history:
            sim_hist = self._max_similarity(feat, self.target.feature_history)

        has_calibrated = self.calibrated.is_ready()
        hist_ok = sim_hist >= self.sim_threshold
        if has_calibrated:
            # Only trust the check when the strict calibrated anchor ALSO
            # agrees; a history-only match is self-referential.
            cal_ok = sim_cal >= self.calibrated_sim_threshold
            verified = cal_ok and hist_ok
        else:
            verified = hist_ok

        return verified, sim_hist, sim_cal, feat

    def _compute_iou(self, a, b) -> float:
        """Intersection over Union for two (x1,y1,x2,y2) boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        xi1 = max(ax1, bx1)
        yi1 = max(ay1, by1)
        xi2 = min(ax2, bx2)
        yi2 = min(ay2, by2)
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        union = a_area + b_area - inter
        return inter / union if union > 0 else 0.0

    def _check_overlap(self, detections: sv.Detections,
                       xyxy=None, tid=None) -> bool:
        """Check whether the followed box overlaps significantly with any
        other tracked detection."""
        if detections.tracker_id is None or len(detections) < 2:
            return False
        if xyxy is None:
            xyxy = self.target.last_xyxy
        if xyxy is None:
            return False
        for i in range(len(detections)):
            if tid is not None and int(detections.tracker_id[i]) == tid:
                continue
            iou = self._compute_iou(xyxy, detections.xyxy[i])
            if iou >= self.overlap_iou_threshold:
                return True
        return False

    # -- helpers -------------------------------------------------------------

    def _embedding_for(self, tid: int, frame: np.ndarray,
                       xyxy) -> np.ndarray | None:
        """Embedding for a candidate, cached per ByteTrack ID with a short TTL
        so per-frame scoring stays affordable on CPU."""
        now = time.time()
        cached = self._embed_cache.get(tid)
        if cached is not None and now - cached[0] < self.embed_cache_ttl:
            return cached[1]
        feat = self._extract_feature(frame, xyxy)
        if feat is None:
            return None
        self._embed_cache[tid] = (now, feat)
        return feat

    def _extract_feature(self, frame: np.ndarray, xyxy) -> np.ndarray | None:
        """Crop the box and run ReID, or ``None`` if the crop is too small."""
        crop = self._crop(frame, xyxy)
        if crop is None or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None
        return self.reid.extract(crop)

    def _max_similarity(self, feat: np.ndarray, refs) -> float:
        if not refs:
            return 0.0
        return float(np.max([self.reid.similarity(feat, r) for r in refs]))

    def _append_feature(self, frame: np.ndarray, xyxy,
                        feat: np.ndarray | None = None) -> None:
        if feat is None:
            feat = self._extract_feature(frame, xyxy)
            if feat is None:
                return
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

    @staticmethod
    def _center(xyxy) -> tuple[float, float]:
        return ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)

    @staticmethod
    def _area(xyxy) -> float:
        return max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1])

    @staticmethod
    def _average_bboxes(bboxes: list[tuple]) -> tuple:
        """Return the per-corner mean of a list of (x1, y1, x2, y2) boxes."""
        mx1, my1, mx2, my2 = (0,0,0,0)
        w = 1.0/len(bboxes)
        for x1, y1, x2, y2 in bboxes:
            mx1 += x1 * w
            my1 += y1 * w
            mx2 += x2 * w
            my2 += y2 * w
        return (mx1, my1, mx2, my2)
