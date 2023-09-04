#!/usr/bin/env python3

"""Exports training/evaluation/test datasets to YOLO format."""

import os
import pathlib
import re
import time

from PIL import Image


def train_test_split(base: str) -> str:
    """Splits dataset into train/eval/test parts deterministically.

    Keep images captured within the same second together in the same
    train/eval set. Otherwise most eval images are extremely similar
    to training images captured a few milliseconds earlier or later,
    which produces high precision & recall metrics but is cheating,
    since we want our model to generalize rather than memorize.
    """
    t = time.strptime(base[: 8 + 1 + 6], "%Y%m%d_%H%M%S")
    remainder = t.tm_sec % 10
    if remainder == 8:
        return "val2023"
    if remainder == 9:
        return "test2023"
    return "train2023"


def path_to_class(path: pathlib.Path) -> str:
    """Converts a directory path to a human-readable sorter class name.

    The class name cannot have slashes in it, to avoid nested
    subdirectories in YOLO images and labels dataset directories.
    """
    if path.name[:4].isdigit():
        # Example: 3702_technic_brick_1x8_with_holes
        return path.name
    # Example: minifig_minecraft_head
    return "_".join(path.parts)


def find_paths_with_jpg_files(path: pathlib.Path) -> list[pathlib.Path]:
    """Finds all subdirectories of working dire that contain JPG images."""
    results = []
    found_jpg = False
    for child in path.iterdir():
        if child.is_dir():
            results.extend(find_paths_with_jpg_files(child))
        elif child.is_file() and child.name.endswith(".jpg"):
            found_jpg = True
    if found_jpg:
        results.append(path)
    return results


def write_yaml(class_names: list[str]):
    """Writes dataset config file in YAML format for YOLO."""
    with open("../yolov7/data/bricks.yaml", "wt", encoding="utf-8") as outfile:
        print("# Bricks dataset by Johann C. Rocholl", file=outfile)
        print("train: bricks/images/train2023", file=outfile)
        print("val: bricks/images/val2023", file=outfile)
        print("test: bricks/images/test2023", file=outfile)
        print(f"nc: {len(class_names)}", file=outfile)
        print("names: [", file=outfile)
        for name in class_names:
            print(f"    {name},", file=outfile)
        print("]", file=outfile)


def export_dir(
    path: pathlib.Path,
    class_id: int,
):
    """Exports one class (directory) of images to YOLO format."""
    class_name = path_to_class(path)
    print(f"id={class_id}\tname={class_name}\tfrom {path}")
    for child in path.iterdir():
        if not child.name.endswith(".jpg"):
            continue
        if not child.is_file():
            continue
        base = re.search(r"^\d{8}_\d{9}", child.name).group(0)
        l = int(re.search(r"_l(\d+)_", child.name).group(1))
        r = int(re.search(r"_r(\d+)_", child.name).group(1))
        t = int(re.search(r"_t(\d+)_", child.name).group(1))
        b = int(re.search(r"_b(\d+)_", child.name).group(1))

        image = Image.open(child)  # This should not read raster data.
        w = image.width
        h = image.height
        assert w * h == 640 * 480

        fx = (l + r) / 2 / w
        fy = (t + b) / 2 / h
        fw = (r - l) / w
        fh = (b - t) / h
        output_base = f"{base}_{class_name}"
        split = train_test_split(base)

        images = pathlib.Path("../yolov7/bricks/images") / split
        output_jpg = images / f"{output_base}.jpg"
        if not output_jpg.exists():
            images.mkdir(parents=True, exist_ok=True)
            os.link(child, output_jpg)

        labels = pathlib.Path("../yolov7/bricks/labels") / split
        labels.mkdir(parents=True, exist_ok=True)
        output_txt = labels / f"{output_base}.txt"
        with open(output_txt, "wt", encoding="utf-8") as txt:
            txt.write(f"{class_id:d} {fx:.3f} {fy:.3f} {fw:.3f} {fh:.3f}\n")


def main():
    """Runs the whole YOLO dataset export for all classes."""
    here = pathlib.Path(".")
    paths = find_paths_with_jpg_files(here)
    paths.sort()
    write_yaml([path_to_class(path) for path in paths])
    for index, path in enumerate(paths):
        export_dir(path, index)


if __name__ == "__main__":
    main()
