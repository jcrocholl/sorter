#!/usr/bin/env python3

import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from yolov9.models.common import Conv
from yolov9.utils.plots import plot_images
from brick_camera import BrickCamera


class ELAN1(torch.nn.Module):
    # elan-1
    def __init__(
        self, c1, c2, c3, c4
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
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


def main() -> None:
    weights_path = "yolov9-s2.pt"
    if not Path(weights_path).exists():
        print(f"Error: {weights_path} not found.")
        return

    print(f"Loading model from {weights_path}...")

    # yolov9 models often require loading custom objects (like numpy arrays)
    # that are now restricted by default in PyTorch 2.6+.
    # We monkeypatch torch.load to default to weights_only=False for this call.
    import functools

    # Monkeypatch missing ELAN1 class into yolov9.models.common
    import yolov9.models.common
    import sys
    import PIL.ImageFont

    yolov9.models.common.ELAN1 = ELAN1  # pyrefly: ignore [missing-attribute]
    # The yolov9 package hacks sys.path, so torch.load may look for 'models.common'
    sys.modules["models.common"] = yolov9.models.common

    # Pillow 10+ removed getsize, which yolov9 still depends on.
    if not hasattr(PIL.ImageFont.FreeTypeFont, "getsize"):

        def getsize(self, text, *args, **kwargs):
            bbox = self.getbbox(text, *args, **kwargs)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])

        PIL.ImageFont.FreeTypeFont.getsize = getsize

    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    try:
        # Determine best device.
        device = "cpu"
        if torch.cuda.is_available():
            device = "0"
        # TODO: Bounding boxes appear to be corrupted on Mac?
        # elif torch.backends.mps.is_available():
        #     device = "mps"

        print(f"Loading model to {device}...")
        model = yolov9.load(
            weights_path,
            device=device,
            # size=640,  # Same as training.
            # half=True,  # Use half-precision for speed.
        )
        # TODO: Bounding boxes appear to be corrupted on Mac?
        # elif torch.backends.mps.is_available():
        #     print("Moving model to MPS...")
        #     model = model.to("mps")
    finally:
        torch.load = original_load

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

        # Run recognition.
        hypotheses = camera.recognize(img, capture_timestamp)

        # Check if the recognized class matches the example subdir name.
        ground_truth = path.parent.name
        recognized_names = [h.class_name for h in hypotheses]
        is_correct = ground_truth in recognized_names
        if is_correct:
            correct_count += 1

        match_status = "OK" if is_correct else "FAIL"
        print(f"  {path.parent.name}/{path.name}: {recognized_names} {match_status}")

        # Collect targets for plot_images: [batch_id, class_id, x, y, w, h] (normalized)
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

        # plot_images expects BCHW and RGB, img is HWC BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_reshaped = img_rgb.transpose(2, 0, 1)  # HWC to CHW
        all_images.append(img_reshaped)
        processed_count += 1

    if not all_images:
        print("No images successfully processed.")
        return

    # Print summary results.
    accuracy = correct_count / processed_count if processed_count > 0 else 0
    print(
        f"\nClassification accuracy: {accuracy:.1%} ({correct_count}/{processed_count})"
    )

    # Print latency results.
    l_min, l_max, l_avg, l_med = camera.latency()
    print("\nInference Latency (seconds):")
    print(f"  Min:    {l_min:.4f}")
    print(f"  Max:    {l_max:.4f}")
    print(f"  Avg:    {l_avg:.4f}")
    print(f"  Median: {l_med:.4f}")

    # Create mosaic grid.
    print("\nCreating mosaic grid...")
    images_np = np.stack(all_images)
    targets_np = np.array(all_targets) if all_targets else np.zeros((0, 6))

    # plot_images is threaded and saves to fname. It does not return the array.
    mosaic_path = "../mosaic.jpg"
    thread = plot_images(
        images=images_np,
        targets=targets_np,
        names=model.names,
        fname=mosaic_path,
    )
    thread.join()

    # Load the resulting mosaic image for display.
    mosaic_bgr = cv2.imread(mosaic_path)
    if mosaic_bgr is None:
        print(f"Error: Failed to read mosaic from {mosaic_path}")
        return

    print(f"Mosaic saved to {mosaic_path}")

    cv2.imshow("BrickCamera Mosaic", mosaic_bgr)
    print("Mosaic displayed. Press any key to quit.")
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
