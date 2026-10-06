from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    """Thin wrapper around Deep-SORT for persistent person IDs."""

    def __init__(self):
        self.object_tracker = DeepSort(
            max_age=20,
            n_init=2,
            nms_max_overlap=0.3,
            max_cosine_distance=0.8,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=False,
            bgr=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
        )

    def track(self, detections, frame):
        tracks = self.object_tracker.update_tracks(detections, frame=frame)

        tracking_ids = []
        boxes = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            tracking_ids.append(track.track_id)
            boxes.append(track.to_ltrb())

        return tracking_ids, boxes
