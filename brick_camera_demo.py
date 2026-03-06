#!/usr/bin/env -S uv run

import argparse
import functools
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import PIL.ImageFont
import torch
from brick_camera import BrickCamera


class ELAN1(torch.nn.Module):
    # elan-1 architecture component for YOLOv9
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        from yolov9.models.common import Conv

        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


def patch_pil() -> None:
    # Pillow 10+ removed getsize, which yolov9 still depends on.
    if not hasattr(PIL.ImageFont.FreeTypeFont, "getsize"):

        def getsize(self, text, *args, **kwargs):
            bbox = self.getbbox(text, *args, **kwargs)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])

        PIL.ImageFont.FreeTypeFont.getsize = getsize


def main() -> None:
    parser = argparse.ArgumentParser(description="Brick Camera Demo")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g., cpu, 0, mps)",
    )
    parser.add_argument(
        "weights",
        type=str,
        nargs="?",
        default="yolov7-tiny.pt",
        help="Path to weights file (e.g., yolov7-tiny.pt or yolov9-s2.pt)",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: {weights_path} not found.")
        return

    is_yolov9 = "yolov9" in weights_path.name.lower()
    print(f"Loading {'YOLOv9' if is_yolov9 else 'YOLOv7'} model from {weights_path}...")
    if is_yolov9:
        import yolov9
        import yolov9.models.common

        patch_pil()
        # pyrefly: ignore[missing-attribute]
        yolov9.models.common.ELAN1 = ELAN1
        sys.modules["models.common"] = yolov9.models.common
        model_loader = yolov9.load
    else:
        import yolov7

        model_loader = yolov7.load

    # yolov7/v9 models often require loading custom objects restricted in PyTorch 2.6+.
    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    try:
        device = "cpu" if args.device == "mps" else args.device
        print(f"Loading to device={device}...")
        model = model_loader(str(weights_path), device=device)
    finally:
        torch.load = original_load

    if args.device == "mps":
        assert torch.backends.mps.is_available(), "MPS not available"
        print("Moving model to MPS...")
        model = model.to("mps")

    # Path to validation images.
    images_root = Path("../exported/images/val2023")
    if not images_root.exists():
        print(f"Error: {images_root} not found.")
        return

    # Collect one JPG image from each class subdirectory with correct dimensions.
    image_paths: list[Path] = []
    target_shape = (640, 480)
    for class_dir in sorted(images_root.iterdir()):
        if class_dir.is_dir():
            jpgs = list(class_dir.glob("*.jpg"))
            random.shuffle(jpgs)
            for jpg_path in jpgs:
                img = cv2.imread(str(jpg_path))
                if img is not None and img.shape[:2] == target_shape:
                    image_paths.append(jpg_path)
                    break

    if not image_paths:
        print(f"No valid {target_shape} images found in {images_root}")
        return

    print(f"Processing {len(image_paths)} images...")

    all_images = []
    all_targets = []
    processed_count = 0
    correct_count = 0

    camera = BrickCamera(model)
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"Failed to load {path}")
            continue

        capture_timestamp = time.time()
        hypotheses = camera.recognize(img, capture_timestamp)

        ground_truth = path.parent.name
        recognized_names = [h.class_name for h in hypotheses]
        is_correct = ground_truth in recognized_names
        if is_correct:
            correct_count += 1

        match_status = "OK" if is_correct else "FAIL"
        print(f"  {path.parent.name}/{path.name}: {recognized_names} {match_status}")

        for hypo in hypotheses:
            all_targets.append(
                [
                    processed_count,
                    hypo.class_id,
                    hypo.x_center,
                    hypo.y_center,
                    hypo.width,
                    hypo.height,
                ]
            )

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_reshaped = img_rgb.transpose(2, 0, 1)
        all_images.append(img_reshaped)
        processed_count += 1

    if not all_images:
        print("No images successfully processed.")
        return

    accuracy = correct_count / processed_count if processed_count > 0 else 0
    print(
        f"\nClassification accuracy: {accuracy:.1%} ({correct_count}/{processed_count})"
    )

    l_min, l_max, l_avg, l_med = camera.latency()
    print("\nInference Latency:")
    print(f"  Min:    {l_min:.4f}s ({1 / l_min:.1f} FPS)")
    print(f"  Max:    {l_max:.4f}s ({1 / l_max:.1f} FPS)")
    print(f"  Avg:    {l_avg:.4f}s ({1 / l_avg:.1f} FPS)")
    print(f"  Median: {l_med:.4f}s ({1 / l_med:.1f} FPS)")

    print("\nCreating mosaic grid...")
    images_np = np.stack(all_images)
    targets_np = np.array(all_targets) if all_targets else np.zeros((0, 6))
    mosaic_path = "../mosaic.jpg"

    if is_yolov9:
        from yolov9.utils.plots import plot_images
    else:
        from yolov7.utils.plots import plot_images

    result = plot_images(
        images=images_np,
        targets=targets_np,
        names=model.names,
        fname=mosaic_path,
    )
    if hasattr(result, "join"):
        result.join()
    print(f"Mosaic saved to {mosaic_path}")


if __name__ == "__main__":
    main()
