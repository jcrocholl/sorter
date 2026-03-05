#!/usr/bin/env python3

import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yolov7
from yolov7.utils.plots import plot_images
from brick_camera import BrickCamera


def main() -> None:
    # Load the real YOLOv7 weights.
    weights_path = "yolov7-tiny.pt"
    if not Path(weights_path).exists():
        print(f"Error: {weights_path} not found.")
        return

    print(f"Loading model from {weights_path}...")

    # yolov7 models often require loading custom objects (like numpy arrays)
    # that are now restricted by default in PyTorch 2.6+.
    # We monkeypatch torch.load to default to weights_only=False for this call.
    import functools

    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    try:
        # Load to CPU first to avoid float64 issues on MPS during torch.load
        model = yolov7.load(
            weights_path,
            device="cpu",  # Cannot load to MPS directly.
            size=640,  # Same as training.
            # TODO: Half precision inference latency > 1.8 seconds per image.
            # half=True,  # Use half-precision for speed.
        )
        # TODO: Bounding boxes appear to be corrupted on Mac?
        # if torch.backends.mps.is_available():
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

    # plot_images returns the mosaic as an RGB numpy array.
    mosaic = plot_images(
        images=images_np,
        targets=targets_np,
        names=model.names,
    )

    # Convert back to BGR for OpenCV display.
    mosaic_bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)

    # Save the resulting mosaic image.
    mosaic_path = "../mosaic.jpg"
    cv2.imwrite(mosaic_path, mosaic_bgr)
    print(f"Mosaic saved to {mosaic_path}")

    cv2.imshow("BrickCamera Mosaic", mosaic_bgr)
    print("Mosaic displayed. Press any key to quit.")
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
