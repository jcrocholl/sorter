#!/usr/bin/env python3

"""Cluster images into groups based on time gaps between them."""

import random
import re
import sys
from pathlib import Path
from datetime import datetime
from yolo_exporter import YoloExporter


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
) -> list[list[tuple[datetime, Path]]]:
    filenames = sorted(input_path.glob("*.jpg"))
    if not filenames:
        return []

    # Parse timestamps and store with original filename
    data = []
    for path in filenames:
        try:
            dt = parse_timestamp(path.name)
            data.append((dt, path))
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

    total_clusters = len(clusters)
    if total_clusters == 0:
        print("No images found to cluster.")
        return

    total_images = sum(len(c) for c in clusters)
    print(f"Total Clusters: {total_clusters}")
    print(f"Total Images:   {total_images}")
    print("-" * 50)

    # Initialize YoloExporter
    exporter = YoloExporter(Path("../yolo_dataset"))

    # Allocate clusters to sets and call YoloExporter
    val_target = int(total_clusters * 0.25)
    test_target = int(total_clusters * 0.25)

    sets = {"train": [], "val": [], "test": []}
    current_cluster_idx = 0

    for set_name, target in [
        ("val", val_target),
        ("test", test_target),
        ("train", total_clusters - val_target - test_target),
    ]:
        for _ in range(target):
            if current_cluster_idx >= total_clusters:
                break
            cluster = clusters[current_cluster_idx]
            sets[set_name].append(cluster)

            # Export first and last file of each cluster
            # First file
            exporter.export_file(
                cluster[0][1], cluster[0][1].parent.name, 0, f"{set_name}2023"
            )
            # Last file (if different from first)
            if len(cluster) > 1:
                exporter.export_file(
                    cluster[-1][1], cluster[-1][1].parent.name, 0, f"{set_name}2023"
                )

            current_cluster_idx += 1

    for set_name in ["train", "val", "test"]:
        cluster_list = sets[set_name]
        image_count = sum(len(c) for c in cluster_list)
        percent = len(cluster_list) / total_clusters * 100 if total_clusters else 0
        print(
            f"{set_name.capitalize():<8} {len(cluster_list):>3} clusters "
            f"({percent:>5.1f}%), {image_count:>4} images"
        )

    exporter.write_yaml()


if __name__ == "__main__":
    main(sys.argv)
