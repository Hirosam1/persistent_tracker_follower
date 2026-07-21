import cv2
import numpy as np
import supervision as sv

from scripts.target_manager import TargetManager, TargetState


GREEN = sv.Color.GREEN
YELLOW = sv.Color.YELLOW
BLUE = sv.Color.BLUE
RED = sv.Color.RED
WHITE = sv.Color(200, 200, 200)


class SceneAnnotator:
    """
    Draws all visual elements — detection boxes, traces, labels,
    target highlight, and HUD — onto each frame.
    """

    def __init__(self) -> None:
        self.box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        self.label_annotator = sv.LabelAnnotator(
            color_lookup=sv.ColorLookup.TRACK,
            smart_position=True,
        )
        self.trace_annotator = sv.TraceAnnotator(
            color_lookup=sv.ColorLookup.TRACK,
        )

    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
        class_names: dict[int, str],
        target_mgr: TargetManager | None,
    ) -> np.ndarray:
        annotated = scene.copy()

        if len(detections) == 0:
            return annotated

        # -- annotation pass (supervision) -----------------------------------
        labels = self._build_labels(detections, class_names, target_mgr)
        target_indices = self._target_indices(detections, target_mgr)

        annotated = self.trace_annotator.annotate(
            scene=annotated, detections=detections,
        )
        annotated = self.box_annotator.annotate(
            scene=annotated, detections=detections,
        )
        annotated = self.label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels,
        )

        # -- target overlay (cv2 direct — avoids sv.Detections API quirks) --
        for i in target_indices:
            x1, y1, x2, y2 = map(int, detections.xyxy[i])

            if target_mgr is not None:
                color = GREEN if target_mgr.state == TargetState.TRACKING else YELLOW
            else:
                color = GREEN

            self._draw_labeled_box(annotated, x1, y1, x2, y2,
                                   f"TARGET #{int(detections.tracker_id[i])}",
                                   color)

        return annotated

    def draw_hud(
        self,
        scene: np.ndarray,
        tracker_name: str,
        target_mgr: TargetManager | None,
        fps: float | None = None,
    ) -> np.ndarray:
        h = scene.shape[0]

        if target_mgr is not None:
            state_name = target_mgr.state.name
            if target_mgr.state == TargetState.TRACKING:
                status_color = GREEN
            elif target_mgr.state == TargetState.SEARCHING:
                status_color = BLUE
            else:
                status_color = RED
        else:
            state_name = "N/A"
            status_color = RED

        lines = [
            f"Tracker: {tracker_name.upper()}",
            f"State: {state_name}",
        ]
        if target_mgr is not None:
            lines.append(f"Target ID: {target_mgr.track_id}")
            if target_mgr.target.last_xyxy is not None:
                IMG_WIDTH=639.0
                CAMERA_FOV_H=np.deg2rad(47.5)/2.0
                x1, y1, x2, y2 = target_mgr.target.last_xyxy
                target_x_center_norm = ((x2-x1)/2+x1)/IMG_WIDTH
                target_angle = (2*CAMERA_FOV_H*target_x_center_norm)-(CAMERA_FOV_H)
                lines.append(f"Angle: {np.rad2deg(target_angle):.1f}")

        for li, txt in enumerate(lines):
            cv2_put(scene, txt, (12, 30 + li * 26), status_color)

        if fps is not None:
            cv2_put(scene, f"FPS: {fps:.1f}", (12, h - 16), WHITE)

        return scene

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _build_labels(
        detections: sv.Detections,
        class_names: dict[int, str],
        target_mgr: TargetManager | None,
    ) -> list[str]:
        labels = []
        for class_id, tid in zip(detections.class_id, detections.tracker_id):
            tid_int = int(tid)
            is_target = (
                target_mgr is not None
                and tid_int == target_mgr.track_id
                and target_mgr.state == TargetState.TRACKING
            )
            if is_target:
                labels.append(f"\u2605 TARGET #{tid_int}")
            else:
                labels.append(f"#{tid_int} {class_names[int(class_id)]}")
        return labels

    @staticmethod
    def _target_indices(
        detections: sv.Detections,
        target_mgr: TargetManager | None,
    ) -> list[int]:
        if target_mgr is None:
            return []
        return [
            i for i, tid in enumerate(detections.tracker_id)
            if int(tid) == target_mgr.track_id
            and target_mgr.state in (TargetState.TRACKING, TargetState.LOST)
        ]

    @staticmethod
    def _draw_labeled_box(
        scene: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        label: str,
        color: sv.Color,
        thickness: int = 3,
    ) -> None:
        bgr = color.as_bgr()
        cv2.rectangle(scene, (x1, y1), (x2, y2), bgr, thickness)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(scene, (x1, y1 - lh - 6), (x1 + lw + 6, y1), bgr, -1)
        cv2.putText(scene, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# bare cv2.putText wrapper to avoid repeating the font args everywhere
def cv2_put(img, text, org, color: sv.Color, scale=0.6, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color.as_bgr(), thick)
