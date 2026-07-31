import os
from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory
    _pkg_dir = get_package_share_directory('persistent_tracker')
except Exception:
    _pkg_dir = Path(__file__).resolve().parent.parent

#MODEL_PATH = os.path.join(_pkg_dir, 'models', 'yolo11n.pt')
# Model generated with `yolo export model=yolo11n.pt format=openvino imgsz=432 half=true`
#MODEL_PATH = os.path.join(_pkg_dir, 'models', 'yolo11n_openvino_model')
# Model generated with `yolo export model=yolo11n.pt format=openvino imgsz=432 quantize=8`
MODEL_PATH = os.path.join(_pkg_dir, 'models', 'yolo11n_int8_openvino_model')
#https://huggingface.co/spaces/rachana219/MODT2/tree/64b1104b0ef5a2ec617e8380e7340f1dc0582fd2/trackers/strongsort/deep/checkpoint
PATH_TO_OSNET_MODEL = os.path.join(_pkg_dir, 'models', 'osnet_x0_25_msmt17.pth')
PATH_TO_OSNET_OPENVINO_XML_MODEL = os.path.join(_pkg_dir, 'models', 'osnet_x1_0_openvino_model/osnet_x1_0_openvino.xml')

# ReID appearance matching
REID_SIMILARITY_THRESHOLD = 0.45
REID_FEATURE_HISTORY_SIZE = 10
REID_SEARCH_EXPAND_RATIO = 2.0
REID_CALIBRATED_SIM_THRESHOLD = 0.7
REID_USE_CALIBRATED_ONLY = True

# Frame pipeline
DEFAULT_CONFIDENCE = 0.5
DEFAULT_TRACKER = "bytetrack"
DEFAULT_VIDEO_SOURCE = "2"

# Tracking confidence
REID_VERIFICATION_INTERVAL = 2.0
CONFIDENCE_BOOST_CALIBRATED = 0.15
CONFIDENCE_BOOST_HISTORY = 0.05
CONFIDENCE_LOST_DECAY_RATE = 0.10
CONFIDENCE_REACQUIRE_PENALTY = 0.25
OVERLAP_IOU_THRESHOLD = 0.3
OVERLAP_PENALTY = 0.15

#Debug
CREATE_DEBUG_IMGS=False
DEBUG_IMGS_FPS=3
DEBUG_RESIZE_FACTOR=0.6
DEBUG_FOLDER="debug_imgs"

