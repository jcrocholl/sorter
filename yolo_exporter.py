#!/usr/bin/env python3

"""Exports training/evaluation/test datasets to YOLO format."""

from collections import defaultdict
import os
import pathlib
import re
from PIL import Image


class YoloExporter:
    """Class to export dataset to YOLO format."""

    def __init__(
        self,
        input_dir: pathlib.Path,
        output_dir: pathlib.Path,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.class_names = []
        self.num_train_per_class = defaultdict(int)
        self.num_val_per_class = defaultdict(int)
        self.num_test_per_class = defaultdict(int)

    def path_to_class(self, path: pathlib.Path) -> str:
        """Converts a directory path to a human-readable sorter class name.

        The class name cannot have slashes in it, to avoid nested
        subdirectories in YOLO images and labels dataset directories.
        """
        try:
            rel_path = path.relative_to(self.input_dir)
        except ValueError:
            return path.name

        if rel_path.parts and rel_path.parts[0] in ("broken", "dirty", "reject"):
            # Examples:
            # broken/98282_vehicle_mudguard_4x2.5x1_with_arch_round_broken
            # dirty/13349_wedge_4x4_triple_inverted_dirty
            # reject/nerf_dart
            return rel_path.parts[0]

        if path.name[:4].isdigit():
            # Example: 3702_technic_brick_1x8_with_holes
            return path.name

        # Example: minifig_minecraft_head
        return "_".join(rel_path.parts)

    def write_yaml(self):
        """Writes dataset config file in YAML format for YOLO."""
        if not self.class_names:
            return
        longest = max(len(class_name) for class_name in self.class_names)
        yaml_file = self.output_dir / "bricks.yaml"
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

        date_match = re.search(r"^\d{8}_\d{9}", child.name)
        if not date_match:
            print(f"failed to parse date_time numbers from {child.name}")
            return
        base = date_match.group(0)

        l_match = re.search(r"_l(\d+)_", child.name)
        r_match = re.search(r"_r(\d+)_", child.name)
        t_match = re.search(r"_t(\d+)_", child.name)
        b_match = re.search(r"_b(\d+)_", child.name)
        if not (l_match and r_match and t_match and b_match):
            print(f"failed to parse l_r_t_b numbers from {child.name}")
            return
        left = int(l_match.group(1))
        right = int(r_match.group(1))
        top = int(t_match.group(1))
        bottom = int(b_match.group(1))

        with Image.open(child) as image:
            width = image.width
            height = image.height
        assert width * height == 640 * 480

        center_x = (left + right) / 2 / width
        center_y = (top + bottom) / 2 / height
        box_width = (right - left) / width
        box_height = (bottom - top) / height
        output_base = f"{base}_{class_name}"

        images = self.output_dir / "images" / split / class_name
        output_jpg = images / f"{output_base}.jpg"
        if not output_jpg.exists():
            images.mkdir(parents=True, exist_ok=True)
            os.link(child, output_jpg)

        labels = self.output_dir / "labels" / split / class_name
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
