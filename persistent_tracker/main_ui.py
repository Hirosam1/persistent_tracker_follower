import argparse
import os
import time
import warnings

# Silence Qt font warnings (must happen before cv2 import)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import supervision as sv
from ultralytics import YOLO

from config import (
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
from ai.extractor import ReIDExtractor
from ai.trackers import build_tracker, NEEDS_FRAME
from annotator import SceneAnnotator
from target_manager import TargetManager, TargetState

warnings.filterwarnings("ignore", message=".*frame argument.*")

# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------

_selected_point: list[tuple[int, int] | None] = [None]


def _mouse_cb(event: int, x: int, y: int, flags: int, param) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        _selected_point[0] = (x, y)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Follow-me robot with appearance-based ReID"
    )
    p.add_argument("--tracker", "-t", default=DEFAULT_TRACKER,
                   choices=["bytetrack", "botsort", "ocsort", "cbiou"])
    p.add_argument("--source", "-s", default=DEFAULT_VIDEO_SOURCE,
                   help="video source (device index or file path)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE,
                   help="detection confidence threshold")
    p.add_argument("--reid-threshold", type=float,
                   default=REID_SIMILARITY_THRESHOLD,
                   help="cosine-similarity threshold for ReID re-acquisition")
    p.add_argument("--full-search", "-f", action="store_true",
                   help="search entire frame (default: velocity-prior region)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Video source helpers
# ---------------------------------------------------------------------------

def _open_source(raw: str) -> cv2.VideoCapture:
    try:
        src = int(raw)
    except ValueError:
        src = raw
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {raw}")
    return cap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main_ui() -> None:
    args = _parse_args()

    # -- video ---------------------------------------------------------------
    cap = _open_source(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"source={args.source}  {w}x{h}  {fps:.1f} FPS  tracker={args.tracker}")

    # -- models --------------------------------------------------------------
    model = YOLO(MODEL_PATH)
    tracker = build_tracker(args.tracker, fps)
    needs_frame = args.tracker in NEEDS_FRAME

    print("Loading ReID extractor (OSNet-x0.25) ...", end=" ", flush=True)
    try:
        reid = ReIDExtractor()
        print(f"device={reid.device}")
    except Exception as exc:
        print(f"FAILED: {exc}")
        reid = None

    target_mgr = (
        TargetManager(
            reid=reid,
            sim_threshold=args.reid_threshold,
            calibrated_sim_threshold=REID_CALIBRATED_SIM_THRESHOLD,
            feature_history_size=REID_FEATURE_HISTORY_SIZE,
            search_expand_ratio=REID_SEARCH_EXPAND_RATIO,
            full_frame_search=args.full_search,
            use_calibrated_only=REID_USE_CALIBRATED_ONLY,
        )
        if reid is not None
        else None
    )

    # -- annotator -----------------------------------------------------------
    annotator = SceneAnnotator()

    cv2.namedWindow("Follow Me")
    cv2.setMouseCallback("Follow Me", _mouse_cb)

    print("Ready.  Click on a person to select target.  'r' reset.  'q' quit.")

    # -- loop ----------------------------------------------------------------
    dtimes: list[float] = []
    fps_timer = time.perf_counter()
    frame_count = 1

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            print("end of stream")
            break

        # --- detect ---
        results = next(model.predict(
            frame, conf=args.conf, classes=[0], verbose=False, stream=True,
        ))
        detections = sv.Detections.from_ultralytics(results)

        # --- track ---
        detections = tracker.update(
            detections=detections,
            frame=frame if needs_frame else None,
        )

        # --- target manager ---
        if target_mgr is not None:
            target_mgr.update(detections, frame, frame_count)

        # --- mouse click → re-designate target ---
        pt = _selected_point[0]
        if pt is not None and len(detections) > 0:
            px, py = pt
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                if x1 <= px <= x2 and y1 <= py <= y2:
                    if target_mgr is not None:
                        target_mgr.reset()
                        target_mgr.designate(detections, frame, det_idx=i)
                    print(f"target re-designated \u2192 track {detections.tracker_id[i]}")
                    break
            _selected_point[0] = None

        # --- annotate ---
        annotated = annotator.annotate(
            scene=frame,
            detections=detections,
            class_names=model.names,
            target_mgr=target_mgr,
        )

        # --- HUD ---
        if len(dtimes) > 0:
            cur_fps = 1.0 / (sum(dtimes) / len(dtimes))
        else:
            cur_fps = None
        annotated = annotator.draw_hud(
            scene=annotated,
            tracker_name=args.tracker,
            target_mgr=target_mgr,
            fps=cur_fps,
        )

        cv2.imshow("Follow Me", annotated)

        # --- keyboard ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            if target_mgr is not None:
                target_mgr.reset()
            print("target reset \u2014 next person will be auto-selected")
        elif key == ord("f"):
            if target_mgr is not None:
                target_mgr.full_frame_search = not target_mgr.full_frame_search
            print(f"full-frame search: {target_mgr.full_frame_search}")

        # --- FPS bookkeeping ---
        frame_count += 1
        dtimes.append(time.perf_counter() - t0)
        if time.perf_counter() - fps_timer > 2.0:
            fps_timer = time.perf_counter()
            dtimes.clear()


if __name__ == "__main__":
    main_ui()
