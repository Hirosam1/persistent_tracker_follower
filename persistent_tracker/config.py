from pathlib import Path

# YOLO detection model
MODEL_PATH = "yolo11n.pt"

# OSNet ReID model weights
# https://huggingface.co/spaces/rachana219/MODT2/tree/64b1104b0ef5a2ec617e8380e7340f1dc0582fd2/trackers/strongsort/deep/checkpoint
PATH_TO_OSNET_MODEL = "osnet_x0_25_msmt17.pth"

# ReID appearance matching
REID_SIMILARITY_THRESHOLD = 0.35
REID_FEATURE_HISTORY_SIZE = 6
REID_SEARCH_EXPAND_RATIO = 2.0
REID_CALIBRATED_SIM_THRESHOLD = 0.72
REID_USE_CALIBRATED_ONLY = True

# Frame pipeline
DEFAULT_CONFIDENCE = 0.6
DEFAULT_TRACKER = "bytetrack"
DEFAULT_VIDEO_SOURCE = "2"
