import argparse
import time
from pathlib import Path

import cv2

from yolo_detector import YoloDetector
from yolo_tracker import Tracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Person detection and multi-object tracking with YOLOv10 + Deep-SORT."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--model",
        default="yolov10m.pt",
        help="YOLO model path/name. Default: yolov10m.pt",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.2,
        help="Detection confidence threshold. Default: 0.2",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for saving the annotated output video.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without opening a preview window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")

    detector = YoloDetector(model_path=args.model, confidence=args.confidence)
    tracker = Tracker()

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {source}")

    writer = None
    if args.output:
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.perf_counter()
            detections = detector.detect(frame)
            tracking_ids, boxes = tracker.track(detections, frame)

            for tracking_id, bounding_box in zip(tracking_ids, boxes):
                x1, y1, x2, y2 = map(int, bounding_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    str(tracking_id),
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            elapsed = time.perf_counter() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("YOLOv10 + Deep-SORT Tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
