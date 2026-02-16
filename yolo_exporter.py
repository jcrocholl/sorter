#!/usr/bin/env python3

"""Exports training/evaluation/test datasets to YOLO format."""

from collections import defaultdict
import os
import pathlib
import re
import time
from PIL import Image


class YoloExporter:
    """Class to export dataset to YOLO format."""

    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.class_names = []
        self.num_train_per_class = defaultdict(int)
        self.num_val_per_class = defaultdict(int)
        self.num_test_per_class = defaultdict(int)

    def train_test_split(self, child: pathlib.Path) -> str:
        """Splits dataset into train/eval/test parts deterministically.

        Keep images captured within the same second together in the same
        train/eval set. Otherwise most eval images are extremely similar
        to training images captured a few milliseconds earlier or later,
        which produces high precision & recall metrics but is cheating,
        since we want our model to generalize rather than memorize.
        """
        base = child.name
        t = time.strptime(base[: 8 + 1 + 6], "%Y%m%d_%H%M%S")
        remainder = t.tm_sec % 10
        if remainder == 8:
            return "val2023"
        if remainder == 9:
            return "test2023"
        return "train2023"

    def path_to_class(self, path: pathlib.Path) -> str:
        """Converts a directory path to a human-readable sorter class name.

        The class name cannot have slashes in it, to avoid nested
        subdirectories in YOLO images and labels dataset directories.
        """
        if path.parts[0] in ("broken", "dirty", "reject"):
            # Examples:
            # broken/98282_vehicle_mudguard_4x2.5x1_with_arch_round_broken
            # dirty/13349_wedge_4x4_triple_inverted_dirty
            # reject/nerf_dart
            return path.parts[0]
        if path.name[:4].isdigit():
            # Example: 3702_technic_brick_1x8_with_holes
            return path.name
        # Example: minifig_minecraft_head
        parts = [p for p in path.parts if p not in ("/", os.sep)]
        return "_".join(parts)

    def find_paths_with_jpg_files(self, path: pathlib.Path) -> list[pathlib.Path]:
        """Finds all subdirectories of path that contain JPG images."""
        results = []
        found_jpg = False
        for child in path.iterdir():
            if child.is_dir():
                results.extend(self.find_paths_with_jpg_files(child))
            elif child.is_file() and child.name.endswith(".jpg"):
                found_jpg = True
        if found_jpg:
            results.append(path)
        return results

    def write_yaml(self):
        """Writes dataset config file in YAML format for YOLO."""
        if not self.class_names:
            return
        longest = max(len(class_name) for class_name in self.class_names)
        yaml_file = self.output_dir / "data" / "bricks.yaml"
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_file, "wt", encoding="utf-8") as outfile:
            print("# Bricks dataset by Johann C. Rocholl", file=outfile)
            print("train: bricks/images/train2023", file=outfile)
            print("val: bricks/images/val2023", file=outfile)
            print("test: bricks/images/test2023", file=outfile)
            print(f"nc: {len(self.class_names)}", file=outfile)
            print("names: [", file=outfile)
            for class_name in self.class_names:
                num_train = self.num_train_per_class[class_name]
                num_val = self.num_val_per_class[class_name]
                num_test = self.num_test_per_class[class_name]
                ljust = " " * (longest - len(class_name))
                comment = f"# train={num_train:<4} val={num_val:<3} test={num_test}"
                print(f"    {class_name},{ljust}  {comment}", file=outfile)
            print("]", file=outfile)

    def export_dir(self, path: pathlib.Path):
        """Exports one class (directory) of images to YOLO format."""
        class_name = self.path_to_class(path)
        print(f"name={class_name}\tfrom {path}")

        for child in path.iterdir():
            if not child.name.endswith(".jpg"):
                continue
            if not child.is_file():
                continue
            split = self.train_test_split(child)
            self.export_file(child, class_name, split)

    def export_file(
        self,
        child: pathlib.Path,
        class_name: str,
        split: str,
    ) -> None:
        """Exports a single image file with labels to YOLO dataset format."""
        try:
            class_id = self.class_names.index(class_name)
        except ValueError:
            class_id = len(self.class_names)
            self.class_names.append(class_name)

        match = re.search(r"^\d{8}_\d{9}", child.name)
        if not match:
            print("failed to parse date_time numbers")
            return
        base = match.group(0)

        try:
            left = int(re.search(r"_l(\d+)_", child.name).group(1))
            right = int(re.search(r"_r(\d+)_", child.name).group(1))
            top = int(re.search(r"_t(\d+)_", child.name).group(1))
            bottom = int(re.search(r"_b(\d+)_", child.name).group(1))
        except (AttributeError, ValueError):
            print("failed to parse l_r_t_b numbers")
            return

        with Image.open(child) as image:
            width = image.width
            height = image.height
        assert width * height == 640 * 480

        center_x = (left + right) / 2 / width
        center_y = (top + bottom) / 2 / height
        box_width = (right - left) / width
        box_height = (bottom - top) / height
        output_base = f"{base}_{class_name}"

        images = self.output_dir / "bricks" / "images" / split
        output_jpg = images / f"{output_base}.jpg"
        if not output_jpg.exists():
            images.mkdir(parents=True, exist_ok=True)
            os.link(child, output_jpg)

        labels = self.output_dir / "bricks" / "labels" / split
        labels.mkdir(parents=True, exist_ok=True)
        output_txt = labels / f"{output_base}.txt"
        with open(output_txt, "wt", encoding="utf-8") as txt:
            txt.write(
                f"{class_id:d} {center_x:.3f} {center_y:.3f} "
                f"{box_width:.3f} {box_height:.3f}\n"
            )

        if "train" in split:
            self.num_train_per_class[class_name] += 1
        elif "val" in split:
            self.num_val_per_class[class_name] += 1
        elif "test" in split:
            self.num_test_per_class[class_name] += 1
        else:
            print("unexpected split:", split)

    def export_all(self, search_path: pathlib.Path = pathlib.Path(".")):
        """Runs the whole YOLO dataset export for all classes."""
        paths = self.find_paths_with_jpg_files(search_path)
        paths.sort()
        for path in paths:
            self.export_dir(path)
        self.write_yaml()
