#!/usr/bin/env python3

from datetime import datetime
import pathlib
import subprocess
import sys
import time

import cv2

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

arg0, CAM, DIR = sys.argv
DIR = DIR.rstrip("/")
CAM = int(CAM)
print(f"CAM={CAM} DIR={DIR}")
pathlib.Path(DIR).mkdir(parents=True, exist_ok=True)


def should_save(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> bool:
    if width < 20:
        print(f"width={width} too narrow")
        return False
    if width > 0.6 * image_width:
        print(f"width={width} too wide")
        return False
    if height < 20:
        print(f"height={height} too short")
        return False
    if height > 0.6 * image_height:
        print(f"height={height} too long")
        return False
    if y < 60:
        print(f"top={y} too far away")
        return False
    bottom = image_height - height - y
    if bottom < 60:
        print(f"bottom={bottom} too close")
        return False
    c = 100 * (x + width // 2) // image_width
    if not 30 < c < 70:
        print(f"center={c} off center")
        return False
    return True


def v4l2_ctl(cam: int, flags: str):
    cmd = f"/usr/bin/v4l2-ctl --device /dev/video{cam} {flags}".split()
    subprocess.run(cmd, check=True)


def adjust_webcam(cam: int):
    v4l2_ctl(cam, "-c auto_exposure=1 -c exposure_time_absolute=10")


cap = cv2.VideoCapture(CAM)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
timestamps = []
while True:
    success, im = cap.read()
    if not success:
        print("failed to read from VideoCapture")
        break

    count += 1
    if count == 10:
        adjust_webcam(CAM)

    # Bricks should move towards the camera, top to bottom:
    im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image_height, image_width, image_channels = im.shape

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    gray = cv2.max(saturation, 255 - value)

    blur = cv2.blur(gray, (5, 5))
    thresh = cv2.Canny(blur, threshold1=80, threshold2=160)
    x, y, width, height = cv2.boundingRect(thresh)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Press q to quit.
        break

    save = (
        width
        and height
        and should_save(
            x=x,
            y=y,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
        )
    )
    if save:
        now = datetime.now().strftime("%Y%m%d_%H%M%S%f")[: 8 + 1 + 6 + 3]
        filename = (
            f"{DIR}/{now}"
            f"_l{x}_r{x + width}"
            f"_t{y}_b{y + height}"
            f"_w{width}_h{height}.jpg"
        )
        print(filename)
        cv2.imwrite(filename, im)

    # Draw rectangle for on-screen debugging, green means saving to disk.
    color = GREEN if save else BLUE
    cv2.rectangle(im, (x, y), (x + width, y + height), color, 3)

    left = cv2.cvtColor(cv2.vconcat([blur, thresh]), cv2.COLOR_GRAY2BGR)
    right = im
    # Uncomment the following line to make the preview window bigger:
    # right = cv2.resize(
    #     im, (image_width * 2, image_height * 2), cv2.INTER_CUBIC
    # )
    lh, lw, lc = left.shape
    rh, rw, rc = right.shape
    left = cv2.resize(left, (lw * rh // lh, rh), cv2.INTER_CUBIC)
    preview = cv2.hconcat([left, right])
    cv2.imshow(DIR, preview)

    timestamps.append(time.time())
    if len(timestamps) > 10:
        fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        print(f"\rfps={fps:.2f}        ", end="", flush=True)
        while len(timestamps) > 99:
            timestamps.pop(0)

print()  # Add final newline after half-written fps line.
