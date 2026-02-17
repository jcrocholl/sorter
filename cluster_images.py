#!/usr/bin/env python3

"""Cluster images into groups based on time gaps between them."""

from collections import defaultdict
import random
import sys
from pathlib import Path
from datetime import datetime
from yolo_exporter import YoloExporter

MIN_CLUSTERS_PER_CLASS = 20


def parse_timestamp(filename: str) -> datetime:
    """Parses timestamp from filename.

    Example: 20230523_105352366_... ->
      datetime(2023, 5, 23, 10, 53, 52, 366000)
    """
    parts = filename.split("_")
    ts_str = parts[0] + parts[1]  # YYYYmmdd + HHMMSSfff
    return datetime.strptime(ts_str, "%Y%m%d%H%M%S%f")


def cluster_images_in_directory(
    input_dir: Path,
    gap_threshold_seconds: float = 0.5,
) -> list[list[tuple[datetime, Path]]]:
    filenames = sorted(input_dir.glob("*.jpg"))
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


def print_summary(sets: dict[str, list], total_clusters: int) -> None:
    """Prints a summary of clusters and images per set."""
    for set_name in ["train", "val", "test"]:
        cluster_list = sets[set_name]
        image_count = sum(len(c) for c in cluster_list)
        percent = len(cluster_list) / total_clusters * 100 if total_clusters else 0
        print(
            f"{set_name}:",
            f"{len(cluster_list)} clusters ({percent:.1f}%),",
            f"{image_count} images",
        )


def cluster_and_filter_by_class(
    exporter: YoloExporter,
    root_dirs: list[str],
) -> list[list[tuple[datetime, Path]]]:
    """Finds images in root_dirs, clusters them, and filters by class count."""
    clusters_by_class = defaultdict(list)

    for root_dir in root_dirs:
        for class_dir in exporter.find_paths_with_jpg_files(Path(root_dir)):
            class_clusters = cluster_images_in_directory(class_dir)
            class_name = exporter.path_to_class(class_dir)
            clusters_by_class[class_name].extend(class_clusters)

    all_clusters = []
    for class_name, clusters in clusters_by_class.items():
        if len(clusters) < MIN_CLUSTERS_PER_CLASS:
            print(
                f"Skipping {class_name} with only {len(clusters)} clusters "
                f"(min {MIN_CLUSTERS_PER_CLASS})."
            )
            continue
        all_clusters.extend(clusters)

    return all_clusters


def export_cluster(
    exporter: YoloExporter,
    cluster: list[tuple[datetime, Path]],
    class_name: str,
    split: str,
) -> None:
    """Exports image(s) from a cluster."""
    # Baseline: Export the median image from each cluster.
    # TODO: Measure how much recognizer accuracy improves
    # if we export multiple images per cluster.
    exporter.export_file(
        child=cluster[len(cluster) // 2][1],
        class_name=class_name,
        split=split,
    )

    if len(cluster) >= 3 and not split.startswith("train"):
        # Export the first image of the cluster.
        exporter.export_file(
            child=cluster[0][1],
            class_name=class_name,
            split=split,
        )
        # Export the last image of the cluster.
        exporter.export_file(
            child=cluster[-1][1],
            class_name=class_name,
            split=split,
        )


def main(argv: list[str]) -> None:
    # Initialize YoloExporter
    exporter = YoloExporter(
        input_dir=Path("../dataset/nested"),
        output_dir=Path("../yolo_dataset"),
    )
    clusters = cluster_and_filter_by_class(exporter, argv[1:] or ["."])

    # Set seed for reproducibility
    random.seed(42)
    # Shuffle clusters for random allocation
    random.shuffle(clusters)

    total_clusters = len(clusters)
    if total_clusters == 0:
        print("No images found to cluster.")
        return

    total_images = sum(len(c) for c in clusters)
    print(f"total clusters: {total_clusters}")
    print(f"total images: {total_images}")
    print("-" * 50)

    # Allocate clusters to sets and call YoloExporter
    val_target = int(total_clusters * 0.25)
    test_target = int(total_clusters * 0.25)

    sets = {"train": [], "val": [], "test": []}
    current_cluster_idx = 0

    for set_name, target in [
        ("val", val_target),
        ("test", test_target),
        ("train", total_clusters),  # rest
    ]:
        for _ in range(target):
            if current_cluster_idx >= total_clusters:
                break
            cluster = clusters[current_cluster_idx]
            sets[set_name].append(cluster)

            # Resolve class name
            class_name = exporter.path_to_class(cluster[0][1].parent)

            export_cluster(
                exporter=exporter,
                cluster=cluster,
                class_name=class_name,
                split=f"{set_name}2023",
            )
            current_cluster_idx += 1

    exporter.write_yaml()
    print_summary(sets, total_clusters)


if __name__ == "__main__":
    main(sys.argv)
