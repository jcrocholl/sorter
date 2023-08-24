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


def should_save(x: int, y: int, w: int, h: int, iw: int, ih: int) -> bool:
    if w < 20:
        print(f"w={w} too narrow")
        return False
    if w > 0.6 * iw:
        print(f"w={w} too wide")
        return False
    if h < 20:
        print(f"h={h} too short")
        return False
    if h > 0.6 * ih:
        print(f"h={h} too long")
        return False
    if y < 60:
        print(f"top={y} too far away")
        return False
    bottom = ih - h - y
    if bottom < 60:
        print(f"bottom={bottom} too close")
        return False
    c = 100 * (x + w // 2) // iw
    if not 30 < c < 70:
        print(f"c={c} off center")
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
    ih, iw, ic = im.shape

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    gray = cv2.max(saturation, 255 - value)

    blur = cv2.blur(gray, (5, 5))
    thresh = cv2.Canny(blur, threshold1=80, threshold2=160)
    x, y, w, h = cv2.boundingRect(thresh)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Press q to quit.
        break

    save = w and h and should_save(x=x, y=y, w=w, h=h, iw=iw, ih=ih)
    if save:
        now = datetime.now().strftime("%Y%m%d_%H%M%S%f")[: 8 + 1 + 6 + 3]
        filename = f"{DIR}/{now}_l{x}_r{x+w}_t{y}_b{y+h}_w{w}_h{h}.jpg"
        print(filename)
        cv2.imwrite(filename, im)

    # Draw rectangle for on-screen debugging, green means saving to disk.
    cv2.rectangle(im, (x, y), (x + w, y + h), GREEN if save else BLUE, 3)

    left = cv2.cvtColor(cv2.vconcat([blur, thresh]), cv2.COLOR_GRAY2BGR)
    right = im
    # Uncomment the following line to make the preview window bigger:
    # right = cv2.resize(im, (iw * 2, ih * 2), cv2.INTER_CUBIC)
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
