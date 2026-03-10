#!/usr/bin/env -S uv run

import argparse
import functools
import time
import cv2
import torch
import yolov7

from brick_camera import BrickCamera
from brick_mapping import BrickMapping
from conveyor_belt import ConveyorBelt
from servo_shelf import ServoShelf

# PCA9685 controller addresses for the sorting shelf
CONTROLLER_CONFIG = {
    0x40: "A1:A10 B1:B5",  # 10 A flaps + 5 B flaps = 15 servos
    0x41: "B6:B10 C1:C10",  # 5 B flaps + 10 C flaps = 15 servos
    0x42: "D1:D10 E1:E5",  # 10 D flaps + 5 E flaps = 15 servos
    0x43: "E6:E10 F1:F10",  # 5 E flaps + 10 F flaps = 15 servos
    0x44: "G1:G10",  # 10 servos
    0x45: "A0:G0",  # 7 kickers
}

# Distances from the camera to each kicker (mm)
KICKER_DISTANCES = {
    "A0": 440.0,
    "B0": 580.0,
    "C0": 720.0,
    "D0": 860.0,
    "E0": 1000.0,
    "F0": 1140.0,
    "G0": 1280.0,
}

BELT_LENGTH = 3600.0  # mm


def get_pca_factory():
    try:
        import adafruit_pca9685
        import board
        import busio

        i2c = busio.I2C(board.SCL, board.SDA)
        return lambda addr: adafruit_pca9685.PCA9685(i2c, address=addr)
    except (ImportError, ValueError, NotImplementedError):
        print("Warning: Hardware PCA9685 not detected. Using mock for testing.")

        class MockPCA:
            def __init__(self, addr):
                self.addr = addr

            def deinit(self):
                pass

            @property
            def channels(self):
                return [type("MockChannel", (), {"duty_cycle": 0})() for _ in range(16)]

        return MockPCA


def main():
    parser = argparse.ArgumentParser(description="Main Brick Sorter Script")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--weights", type=str, default="yolov7-tiny.pt", help="Path to weights file"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu, mps, 0 for cuda)"
    )
    args = parser.parse_args()

    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    try:
        model = yolov7.load(args.weights, device=args.device)
    finally:
        torch.load = original_load

    camera = BrickCamera(model)
    belt = ConveyorBelt(length=BELT_LENGTH, kicker_distances=KICKER_DISTANCES)
    mapping = BrickMapping("drawers/brick_classes.csv")
    shelf = ServoShelf(
        config=CONTROLLER_CONFIG,
        pca_factory=get_pca_factory(),
        conveyor_belt=belt,
        brick_mapping=mapping,
    )

    shelf.start()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting main loop. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Match webcam.py rotation: bricks move top-to-bottom
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            capture_time = time.time()

            hypotheses = camera.recognize(frame, capture_time)

            for h in hypotheses:
                # Bounding box for debug display
                h_h, h_w = frame.shape[:2]
                x1 = int((h.x_center - h.width / 2) * h_w)
                y1 = int((h.y_center - h.height / 2) * h_h)
                x2 = int((h.x_center + h.width / 2) * h_w)
                y2 = int((h.y_center + h.height / 2) * h_h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{h.class_name} {h.confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                # Special handling for belt marks (using 3005_brick_1x1 as a marker)
                if h.class_name == "3005_brick_1x1":
                    belt.observed_mark_at(h.class_name, capture_time)
                    print(f"Calibration mark seen. Speed: {belt.speed:.1f} mm/s")
                else:
                    try:
                        shelf.on_brick_recognized(capture_time, h.class_name)
                        print(f"Recognized: {h.class_name}")
                    except KeyError:
                        # Skip if class not in mapping
                        pass

            cv2.imshow("ConveyorBelt", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        shelf.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
