#!/usr/bin/env python3

"""Cluster images into groups based on time gaps between them."""

import random
import re
import sys
from pathlib import Path
from datetime import datetime


def parse_yolo_label(filename: str, class_id: int = 0) -> str:
    """Extracts bounding box from filename and formats as YOLO label."""
    # Example filename: 20230511_121859430_l215_r404_t144_b312_w189_h168.jpg
    match = re.search(r"_l(\d+)_r(\d+)_t(\d+)_b(\d+)_", filename)
    if not match:
        return "N/A"

    left, right, top, bottom = map(int, match.groups())

    # Assuming 640x480 as seen in export_yolo.py
    width, height = 640, 480

    center_x = (left + right) / 2 / width
    center_y = (top + bottom) / 2 / height
    box_width = (right - left) / width
    box_height = (bottom - top) / height

    return f"{class_id} {center_x:.3f} {center_y:.3f} {box_width:.3f} {box_height:.3f}"


def parse_timestamp(filename: str) -> datetime:
    """Parses timestamp from filename.

    Example: 20230523_105352366_... ->
      datetime(2023, 5, 23, 10, 53, 52, 366000)
    """
    parts = filename.split("_")
    ts_str = parts[0] + parts[1]  # YYYYmmdd + HHMMSSfff
    return datetime.strptime(ts_str, "%Y%m%d%H%M%S%f")


def cluster_images(
    input_path: Path,
    gap_threshold_seconds: float = 0.5,
) -> list[list[tuple[datetime, str]]]:
    filenames = sorted([p.name for p in input_path.glob("*.jpg")])
    if not filenames:
        return []

    # Parse timestamps and store with original filename
    data = []
    for name in filenames:
        try:
            dt = parse_timestamp(name)
            data.append((dt, name))
        except (ValueError, IndexError):
            continue

    # Sort by datetime
    data.sort()

    clusters = []
    if not data:
        return []

    current_cluster = [data[0]]

    for i in range(1, len(data)):
        prev_dt, _ = data[i - 1]
        curr_dt, name = data[i]

        gap = (curr_dt - prev_dt).total_seconds()

        if gap <= gap_threshold_seconds:
            current_cluster.append(data[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [data[i]]

    clusters.append(current_cluster)
    return clusters


def main(argv: list[str]) -> None:
    # Set seed for reproducibility
    random.seed(42)

    args = argv[1:]
    if not args:
        args = ["."]

    clusters = []
    for arg in args:
        clusters.extend(cluster_images(Path(arg)))

    # Shuffle clusters for random allocation
    random.seed(42)  # Reset seed for stability in cluster order
    random.shuffle(clusters)

    for cluster in clusters:
        first_name = cluster[0][1]
        last_name = cluster[-1][1]
        first_label = parse_yolo_label(first_name)
        last_label = parse_yolo_label(last_name)
        print(f"Cluster: {first_name} {first_label}")
        print(f"         {last_name} {last_label}")

    total_images = sum(len(c) for c in clusters)
    total_clusters = len(clusters)
    if total_clusters == 0:
        print("No images found to cluster.")
        return

    val_target = int(total_clusters * 0.25)
    test_target = int(total_clusters * 0.25)

    train_data = []
    val_data = []
    test_data = []
    for i, cluster in enumerate(clusters):
        if i < val_target:
            val_data.append(cluster)
        elif i < (val_target + test_target):
            test_data.append(cluster)
        else:
            train_data.append(cluster)

    print(f"Total Clusters: {total_clusters}")
    print(f"Total Images:   {total_images}")
    print("-" * 50)
    train_clusters = len(train_data)
    val_clusters = len(val_data)
    test_clusters = len(test_data)
    train_images = sum(len(c) for c in train_data)
    val_images = sum(len(c) for c in val_data)
    test_images = sum(len(c) for c in test_data)
    print(
        f"Train: {train_clusters:3} clusters "
        f"({train_clusters / total_clusters:5.1%}), "
        f"{train_images:4} images"
    )
    print(
        f"Val:   {val_clusters:3} clusters "
        f"({val_clusters / total_clusters:5.1%}), "
        f"{val_images:4} images"
    )
    print(
        f"Test:  {test_clusters:3} clusters "
        f"({test_clusters / total_clusters:5.1%}), "
        f"{test_images:4} images"
    )


if __name__ == "__main__":
    main(sys.argv)
