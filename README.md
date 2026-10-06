# YOLOv10 + Deep-SORT Person Tracking

A computer vision project for **person detection and multi-object tracking in video**. The pipeline uses **YOLOv10** for frame-level person detection and **Deep-SORT** to maintain persistent identities across frames.

## Project overview

For each video frame, the system:

1. detects people with YOLOv10,
2. converts the detections to Deep-SORT format,
3. updates the multi-object tracker,
4. assigns persistent track IDs,
5. draws bounding boxes and IDs, and
6. reports per-frame processing FPS.

The project was originally developed with a YOLOv10 medium model (`yolov10m.pt`) and a MobileNet-based Deep-SORT appearance embedder.

## Tech stack

- Python
- OpenCV
- Ultralytics YOLOv10
- Deep-SORT (`deep-sort-realtime`)
- MobileNet appearance embeddings

## Repository structure

```text
.
├── src/
│   ├── main.py
│   ├── yolo_detector.py
│   └── yolo_tracker.py
├── video/
│   └── README.md
├── docs/
│   └── project_documentation.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
git clone https://github.com/imkiyanaz/yolov10-deepsort-person-tracking.git
cd yolov10-deepsort-person-tracking
pip install -r requirements.txt
```

## Usage

Provide a local input video:

```bash
python src/main.py --source video/football.mp4
```

You can also set a detection confidence threshold and save the annotated result:

```bash
python src/main.py \
  --source video/football.mp4 \
  --model yolov10m.pt \
  --confidence 0.2 \
  --output outputs/tracked.mp4
```

For headless environments:

```bash
python src/main.py --source video/football.mp4 --no-display
```

## Detection and tracking pipeline

### YOLO detector

The detector keeps only the **person** class and outputs bounding boxes in the format expected by `deep-sort-realtime`:

```text
([left, top, width, height], confidence, class_name)
```

### Deep-SORT tracker

Deep-SORT combines motion and appearance information to associate detections over time. Confirmed tracks are returned with persistent IDs and left-top-right-bottom coordinates.

## Code-quality cleanup for the public version

The original coursework implementation has been reorganized for portfolio use. The public version:

- removes compiled `__pycache__` / `.pyc` files,
- excludes local model weights and source videos,
- adds command-line arguments for reusable execution,
- optionally saves annotated output video,
- converts YOLO confidence values to plain Python floats, and
- uses the detection tuple order expected by `deep-sort-realtime` (`bbox`, `confidence`, `class`).

## Current limitations

- The repository does not include a benchmark dataset or quantitative tracking metrics such as MOTA/IDF1.
- Runtime FPS depends strongly on hardware, video resolution, model size, and the number of visible people.
- Model weights and the original test video are intentionally not bundled in the repository.

## Possible extensions

- people counting and line-crossing analytics,
- trajectory visualization and heatmaps,
- sports analytics,
- crowd monitoring,
- GPU-specific performance benchmarking,
- evaluation with MOT metrics such as MOTA and IDF1.

## Documentation

The original project documentation is available in [`docs/project_documentation.pdf`](docs/project_documentation.pdf).
