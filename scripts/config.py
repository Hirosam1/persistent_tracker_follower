import os
from dataclasses import dataclass
from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory
    _pkg_dir = get_package_share_directory('persistent_tracker')
except Exception:
    _pkg_dir = Path(__file__).resolve().parent.parent

#MODEL_PATH = os.path.join(_pkg_dir, 'models', 'yolo11n.pt')
# Model generated with `yolo export model=yolo11n.pt format=openvino imgsz=432 half=true`
#MODEL_PATH = os.path.join(_pkg_dir, 'models', 'yolo11n_openvino_model')
# Model generated with `yolo export model=yolo11n.pt format=openvino imgsz=412 quantize=8`
MODEL_PATH = os.path.join(_pkg_dir, 'models', 'yolo11n_int8_openvino_model')
#https://huggingface.co/spaces/rachana219/MODT2/tree/64b1104b0ef5a2ec617e8380e7340f1dc0582fd2/trackers/strongsort/deep/checkpoint
PATH_TO_OSNET_MODEL = os.path.join(_pkg_dir, 'models', 'osnet_x0_25_msmt17.pth')
PATH_TO_OSNET_OPENVINO_XML_MODEL = os.path.join(_pkg_dir, 'models', 'osnet_x1_0_openvino_model/osnet_x1_0_openvino.xml')

# ReID appearance matching
REID_SIMILARITY_THRESHOLD = 0.45
REID_FEATURE_HISTORY_SIZE = 10
REID_SEARCH_EXPAND_RATIO = 2.0
REID_CALIBRATED_SIM_THRESHOLD = 0.7

# Frame pipeline
DEFAULT_CONFIDENCE = 0.5
DEFAULT_TRACKER = "bytetrack"
DEFAULT_VIDEO_SOURCE = "2"

# Tracking confidence
REID_VERIFICATION_INTERVAL = 2.3
CONFIDENCE_BOOST_CALIBRATED = 0.19
CONFIDENCE_BOOST_HISTORY = 0.05
CONFIDENCE_LOST_DECAY_RATE = 0.7
OVERLAP_IOU_THRESHOLD = 0.3

# Identity management (appearance-first)
# Evidence levels: consecutive confirmations required to adopt a new identity
EVIDENCE_NORMAL_VOTES = 2
EVIDENCE_SUSPICIOUS_VOTES = 3
EVIDENCE_LOCKED_VOTES = 5

# Suspicion thresholds
SUSPICION_EVIDENCE_THRESHOLD = 0.42   # selects the "suspicious" evidence tier
SUSPICION_VERIFY_THRESHOLD = 0.53     # triggers immediate event-driven ReID
SUSPICION_FREEZE_THRESHOLD = 0.4     # freezes feature learning
SUSPICION_DECAY_RATE = 0.17          # per second
SUSPICION_AMBIGUITY_GAP = 0.07       # score gap that counts as ambiguous

# Suspicion event weights
SUSPICION_OVERLAP_WEIGHT = 0.40
SUSPICION_JUMP_WEIGHT = 0.20
SUSPICION_AREA_WEIGHT = 0.12
SUSPICION_ID_MISS_WEIGHT = 0.30
SUSPICION_NEW_PERSON_WEIGHT = 0.15
SUSPICION_CONF_DROP_WEIGHT = 0.20
SUSPICION_AMBIGUITY_WEIGHT = 0.20
SUSPICION_DRIFT_WEIGHT = 0.1

# Suspicious-event detection thresholds
BBOX_JUMP_PIXELS = 110.0
BBOX_AREA_CHANGE_RATIO = 1.6
NEW_PERSON_PROXIMITY_PX = 150.0
CONF_DROP_THRESHOLD = 0.35

# Identity lock
LOCK_CONFIDENCE = 0.7
UNLOCK_CONFIDENCE = 0.4
LOCK_MIN_STABLE_FRAMES = 30
LOCK_LOST_GRACE = 2.0

# Learning freeze / resume
LEARNING_FREEZE_SECONDS = 1.5
REACQ_LEARNING_COOLDOWN = 2.0

# Verification
EVENT_VERIFY_MIN_INTERVAL = 0.5
CONFIDENCE_SUSPICION_PENALTY = 0.10

# Appearance scoring & caching
APPEARANCE_CAL_WEIGHT = 0.7
EMBED_CACHE_TTL = 0.4

#Debug
CREATE_DEBUG_IMGS=False
DEBUG_IMGS_FPS=3
DEBUG_RESIZE_FACTOR=0.6
DEBUG_FOLDER="debug_imgs"


@dataclass
class TargetManagerConfig:
    """All tunables for :class:`TargetManager`. Defaults mirror the constants
    above, so callers only need to override the values that differ."""
    sim_threshold: float = REID_SIMILARITY_THRESHOLD
    calibrated_sim_threshold: float = REID_CALIBRATED_SIM_THRESHOLD
    feature_history_size: int = REID_FEATURE_HISTORY_SIZE
    search_expand_ratio: float = REID_SEARCH_EXPAND_RATIO
    full_frame_search: bool = True

    # -- confidence ---------------------------------------------------
    verification_interval: float = REID_VERIFICATION_INTERVAL
    confidence_boost_calibrated: float = CONFIDENCE_BOOST_CALIBRATED
    confidence_boost_history: float = CONFIDENCE_BOOST_HISTORY
    confidence_lost_decay_rate: float = CONFIDENCE_LOST_DECAY_RATE
    overlap_iou_threshold: float = OVERLAP_IOU_THRESHOLD

    # -- evidence tiers -------------------------------------------------
    evidence_normal_votes: int = EVIDENCE_NORMAL_VOTES
    evidence_suspicious_votes: int = EVIDENCE_SUSPICIOUS_VOTES
    evidence_locked_votes: int = EVIDENCE_LOCKED_VOTES

    # -- suspicion ------------------------------------------------------
    suspicion_evidence_threshold: float = SUSPICION_EVIDENCE_THRESHOLD
    suspicion_verify_threshold: float = SUSPICION_VERIFY_THRESHOLD
    suspicion_freeze_threshold: float = SUSPICION_FREEZE_THRESHOLD
    suspicion_decay_rate: float = SUSPICION_DECAY_RATE
    suspicion_ambiguity_gap: float = SUSPICION_AMBIGUITY_GAP
    suspicion_overlap_weight: float = SUSPICION_OVERLAP_WEIGHT
    suspicion_jump_weight: float = SUSPICION_JUMP_WEIGHT
    suspicion_area_weight: float = SUSPICION_AREA_WEIGHT
    suspicion_id_miss_weight: float = SUSPICION_ID_MISS_WEIGHT
    suspicion_new_person_weight: float = SUSPICION_NEW_PERSON_WEIGHT
    suspicion_conf_drop_weight: float = SUSPICION_CONF_DROP_WEIGHT
    suspicion_ambiguity_weight: float = SUSPICION_AMBIGUITY_WEIGHT
    suspicion_drift_weight: float = SUSPICION_DRIFT_WEIGHT
    bbox_jump_pixels: float = BBOX_JUMP_PIXELS
    bbox_area_change_ratio: float = BBOX_AREA_CHANGE_RATIO
    new_person_proximity_px: float = NEW_PERSON_PROXIMITY_PX
    conf_drop_threshold: float = CONF_DROP_THRESHOLD

    # -- identity lock ---------------------------------------------------
    lock_confidence: float = LOCK_CONFIDENCE
    unlock_confidence: float = UNLOCK_CONFIDENCE
    lock_min_stable_frames: int = LOCK_MIN_STABLE_FRAMES
    lock_lost_grace: float = LOCK_LOST_GRACE

    # -- learning freeze ------------------------------------------------
    learning_freeze_seconds: float = LEARNING_FREEZE_SECONDS
    reacq_learning_cooldown: float = REACQ_LEARNING_COOLDOWN

    # -- verification ---------------------------------------------------
    event_verify_min_interval: float = EVENT_VERIFY_MIN_INTERVAL
    confidence_suspicion_penalty: float = CONFIDENCE_SUSPICION_PENALTY

    # -- scoring / cache ------------------------------------------------
    appearance_cal_weight: float = APPEARANCE_CAL_WEIGHT
    embed_cache_ttl: float = EMBED_CACHE_TTL

