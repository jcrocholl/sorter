from dataclasses import dataclass
from typing import Any

import statistics
import time

import cv2
import cv2.typing


@dataclass
class Hypothesis:
    """A single object detection hypothesis returned by BrickCamera.recognize().

    Bounding box coordinates are in YOLO normalized format (each value is a
    fraction of the total image dimension, between 0.0 and 1.0):
        x_center: horizontal center of the bounding box
        y_center: vertical center of the bounding box
        width:    width of the bounding box
        height:   height of the bounding box
    """

    confidence: float
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int
    class_name: str


class BrickCamera:
    """Runs YOLO object recognition on camera frames to detect brick types."""

    def __init__(self, model: Any) -> None:
        """Initialize with a pre-loaded YOLOv7 model.

        Args:
            model: A loaded YOLOv7 model (e.g. from yolov7.load()) or a
                compatible mock for testing.
        """
        self._model = model
        self._latencies: list[float] = []

    def recognize(
        self,
        img: cv2.typing.MatLike,
        capture_timestamp: float,
    ) -> list[Hypothesis]:
        """Detect bricks in a camera frame and return recognition hypotheses.

        Args:
            img: A BGR image as returned by cv2.VideoCapture.read().
            capture_timestamp: The time (seconds) when this frame was captured.

        Returns:
            A list of Hypothesis objects, one per detected object. May be empty
            if no bricks were detected. Results are ordered by confidence
            (highest first) as determined by the YOLO model's NMS output.
        """
        # yolov7 accepts BGR numpy arrays directly (same format as cv2).
        start_time = time.perf_counter()
        results = self._model(img)
        end_time = time.perf_counter()
        self._latencies.append(end_time - start_time)

        hypotheses: list[Hypothesis] = []
        # results.xywhn is a list of tensors, one per image in the batch.
        # We pass a single image, so we only use index 0.
        for *xywh, conf, cls in results.xywhn[0].tolist():
            x_center, y_center, width, height = xywh
            hypotheses.append(
                Hypothesis(
                    confidence=conf,
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                    class_id=int(cls),
                    class_name=results.names[int(cls)],
                )
            )
        return hypotheses

    def latency(self) -> tuple[float, float, float, float]:
        """Return the (min, max, avg, median) latency of model inference in seconds.

        Returns:
            A tuple of (min, max, average, median) latency in float seconds.
            Returns (0.0, 0.0, 0.0, 0.0) if no inferences have been run.
        """
        if not self._latencies:
            return 0.0, 0.0, 0.0, 0.0

        return (
            min(self._latencies),
            max(self._latencies),
            statistics.mean(self._latencies),
            statistics.median(self._latencies),
        )
