#!/usr/bin/env python3

from hashlib import sha1
import os
import pathlib
import re
import shutil
import struct

import cv2
from PIL import Image

# Don't swap color channels on these.
PRESERVE_COLOR_CHANNELS = [
    "animal",
    "human_tool",
    "minifig",
    "plant",
    "technic_axle",
    "technic_connector",
    "technic_gear",
]


def sha1_percent(key: str) -> float:
    bytes = sha1(key.encode("utf-8")).digest()[:8]
    return float(struct.unpack("L", bytes)[0] * 100) / 2**64


def train_test_split(base: str) -> str:
    percent = sha1_percent(base)
    if percent < 80:
        return "train2023"
    elif percent < 90:
        return "val2023"
    else:
        return "test2023"


def path_to_class(path: pathlib.Path) -> str:
    slash_parts = list(path.parts)
    underscore_parts = slash_parts[-1].split("_")
    if underscore_parts[0][:4].isdigit():
        # Remove the long part name, keep the number.
        slash_parts[-1] = underscore_parts[0]
    return "_".join(slash_parts)


def find_paths_with_jpg_files(path: pathlib.Path) -> list[pathlib.Path]:
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
    with open("../yolov7/data/bricks.yaml", "wt") as outfile:
        print("# Bricks dataset by Johann C. Rocholl", file=outfile)
        print("train: bricks/images/train2023", file=outfile)
        print("val: bricks/images/val2023", file=outfile)
        print("test: bricks/images/test2023", file=outfile)
        print(f"nc: {len(class_names)}", file=outfile)
        print("names: [", file=outfile)
        for name in class_names:
            print(f"    {name},", file=outfile)
        print("]", file=outfile)


def export_jpg(class_name: str, src: pathlib.Path, dst: pathlib.Path):
    percent = int(sha1_percent(str(src.name)))
    if percent & 7 == 0 or any(
        class_name.startswith(prefix) for prefix in PRESERVE_COLOR_CHANNELS
    ):
        # Avoid hard links because then if we do decide to modify
        # it later then that would also modify the source.
        shutil.copy(src, dst)
    else:
        # Deterministically swap blue / green / red color channels.
        img = cv2.imread(str(src))
        b, g, r = cv2.split(img)
        if percent & 1:
            r, g = g, r
        if percent & 2:
            g, b = b, g
        if percent & 4:
            r, b = b, r
        img = cv2.merge([b, g, r])
        cv2.imwrite(str(dst), img)


def export_dir(
    path: pathlib.Path,
    class_id: int,
):
    class_name = path_to_class(path)
    print(f"exporting {path} as name={class_name} id={class_id}")
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
            # Uncomment if you want to swap color channels:
            # export_jpg(class_name, child, output_jpg)

        labels = pathlib.Path("../yolov7/bricks/labels") / split
        labels.mkdir(parents=True, exist_ok=True)
        output_txt = labels / f"{output_base}.txt"
        with open(output_txt, "wt") as txt:
            txt.write(f"{class_id:d} {fx:.3f} {fy:.3f} {fw:.3f} {fh:.3f}\n")


def main():
    here = pathlib.Path(".")
    paths = find_paths_with_jpg_files(here)
    paths.sort()
    write_yaml([path_to_class(path) for path in paths])
    for index, path in enumerate(paths):
        export_dir(path, index)


if __name__ == "__main__":
    main()
