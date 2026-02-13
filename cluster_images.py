#!/usr/bin/env python3
import random
import re
from datetime import datetime


def parse_yolo_label(filename, class_id=0):
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


def cluster_images(file_list_path, gap_threshold_seconds=1):
    with open(file_list_path, "r") as f:
        filenames = [line.strip() for line in f if line.strip().endswith(".jpg")]

    if not filenames:
        return []

    # Parse timestamps and store with original filename
    data = []
    for name in filenames:
        try:
            # Format: 20230523_105352366_...
            parts = name.split("_")
            ts_str = parts[0] + parts[1][:6]  # YYYYMMDD + HHMMSS
            dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
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


def save_list(filename, data):
    with open(filename, "w") as f:
        for cluster in data:
            for dt, name in cluster:
                f.write(f"{name}\n")


def main():
    # Set seed for reproducibility
    random.seed(42)

    path = "/tmp/all_filenames.txt"
    clusters = cluster_images(path)

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

    save_list("train.txt", train_data)
    save_list("val.txt", val_data)
    save_list("test.txt", test_data)


if __name__ == "__main__":
    main()
