import os
from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory
    _pkg_dir = get_package_share_directory('persistent_tracker')
except Exception:
    _pkg_dir = Path(__file__).resolve().parent.parent

#MODEL_PATH = os.path.join(_pkg_dir, 'models', 'yolo11n.pt')
#MODEL_PATH = os.path.join(_pkg_dir, 'models', 'yolo11n_openvino_model')
MODEL_PATH = os.path.join(_pkg_dir, 'models', 'yolo11s_openvino_model')

PATH_TO_OSNET_MODEL = os.path.join(_pkg_dir, 'models', 'osnet_x0_25_msmt17.pth')

# ReID appearance matching
REID_SIMILARITY_THRESHOLD = 0.35
REID_FEATURE_HISTORY_SIZE = 6
REID_SEARCH_EXPAND_RATIO = 2.0
REID_CALIBRATED_SIM_THRESHOLD = 0.70
REID_USE_CALIBRATED_ONLY = True

# Frame pipeline
DEFAULT_CONFIDENCE = 0.6
DEFAULT_TRACKER = "bytetrack"
DEFAULT_VIDEO_SOURCE = "2"
