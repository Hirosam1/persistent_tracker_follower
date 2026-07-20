from trackers import (
    ByteTrackTracker,
    BoTSORTTracker,
    CBIoUTracker,
    OCSORTTracker,
)

NEEDS_FRAME = {"botsort"}


def build_tracker(name: str, fps: float):
    name = name.lower()
    if name == "bytetrack":
        return ByteTrackTracker(
            frame_rate=fps,
            lost_track_buffer=180,
            minimum_iou_threshold=0.10,
            high_conf_det_threshold=0.6,
            track_activation_threshold=0.75,
            minimum_consecutive_frames=8,
        )
    elif name == "botsort":
        return BoTSORTTracker(
            frame_rate=fps,
            lost_track_buffer=240,
            minimum_iou_threshold_first_assoc=0.10,
            minimum_iou_threshold_second_assoc=0.20,
            minimum_iou_threshold_unconfirmed_assoc=0.15,
            high_conf_det_threshold=0.5,
            track_activation_threshold=0.7,
            minimum_consecutive_frames=10,
            enable_cmc=True,
            cmc_method="sparseOptFlow",
            cmc_downscale=2,
        )
    elif name == "ocsort":
        return OCSORTTracker(
            frame_rate=fps,
            lost_track_buffer=180,
            minimum_consecutive_frames=8,
            minimum_iou_threshold=0.20,
            direction_consistency_weight=0.30,
            high_conf_det_threshold=0.5,
            delta_t=5,
        )
    elif name == "cbiou":
        return CBIoUTracker(
            frame_rate=fps,
            lost_track_buffer=240,
            minimum_consecutive_frames=8,
            minimum_iou_threshold_first_assoc=0.10,
            minimum_iou_threshold_second_assoc=0.25,
            minimum_iou_threshold_unconfirmed_assoc=0.15,
            high_conf_det_threshold=0.5,
            track_activation_threshold=0.7,
            buffer_ratio_first=0.3,
            buffer_ratio_second=0.5,
        )
    else:
        raise ValueError(
            f"Unknown tracker: {name}  (options: bytetrack, botsort, ocsort, cbiou)"
        )
