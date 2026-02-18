#!/usr/bin/env python3

from collections import defaultdict
from collections.abc import Iterable
import os
import re
import subprocess
import sys

import cv2

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


class Histogram:
    def __init__(self, name: str, regex: str):
        self.name = name
        self.regex = re.compile(regex)
        self.count = defaultdict(int)
        self.total = 0
        self.min = 1_000_000
        self.max = -1_000_000
        for filename in self.filenames():
            match = self.regex.search(filename)
            if match:
                n = int(match.group(1))
                self.count[n] += 1
                self.total += 1
                if n < self.min:
                    self.min = n
                if n > self.max:
                    self.max = n

    def filenames(self, dirname=".", ext=".jpg") -> Iterable[str]:
        if len(sys.argv) > 1:
            for filename in sys.argv[1:]:
                yield filename
        else:
            for filename in os.listdir(dirname):
                if filename.endswith(ext):
                    yield filename

    def percentile(self, p: int) -> int:
        if p <= 0:
            return self.min
        if p >= 100:
            return self.max
        seen = 0
        for n in range(self.min, self.max + 1):
            if n in self.count:
                seen += self.count[n]
                if seen / p > self.total / 100:
                    return n
        return self.max

    def evaluate(self):
        print("name:", self.name)
        print("total:", self.total)
        print("min:", self.min)
        print("max:", self.max)
        percentile = {}
        for p in (1, 2, 5, 10, 20, 50, 80, 90, 95, 98, 99):
            n = self.percentile(p)
            percentile[n] = p
        for n in range(self.min, self.max + 1):
            if n in self.count:
                print(
                    n,
                    "*" * self.count[n],
                    f"{percentile[n]}%" if n in percentile else "",
                )

    def bbox(self, filename: str) -> str:
        """Returns a temporary image with bounding box."""
        temp = "/tmp/" + os.path.basename(filename)
        im = cv2.imread(filename)
        # Draw rectangle for on-screen debugging.
        match = re.search(r"l(\d+)_r(\d+)_t(\d+)_b(\d+)_w(\d+)_h(\d+)", filename)
        assert match
        left = int(match.group(1))
        right = int(match.group(2))
        top = int(match.group(3))
        bottom = int(match.group(4))
        width = int(match.group(5))
        height = int(match.group(6))
        assert width == right - left
        assert height == bottom - top
        cv2.rectangle(im, (left, top), (right, bottom), BLUE, 3)
        cv2.imwrite(temp, im)
        return temp

    def feh(self, filenames: list[str]):
        """Human review for outliers, use C-Del to remove bad examples."""
        feh = [
            "/usr/bin/feh",
            "--auto-zoom",
            "--g",
            "2000x2000",
            "--on-last-slide",
            "quit",
        ]
        pairs = [
            (original, self.bbox(original))
            for original in filenames
            if os.path.exists(original)  # It may have been deleted.
        ]
        feh.extend([bbox for (_, bbox) in pairs])
        subprocess.run(feh, check=True)
        num_deleted = 0
        for original, bbox in pairs:
            if not os.path.exists(bbox):
                print(f"{bbox} was deleted => deleting {original}")
                os.remove(original)
                num_deleted += 1
        if num_deleted:
            print(f"deleted {num_deleted} of {len(pairs)} files")

    def review(self, lo: int, hi: int):
        self.evaluate()
        n_lo = self.percentile(lo)
        n_hi = self.percentile(hi)
        pairs_lo = []
        pairs_hi = []
        for filename in self.filenames():
            match = self.regex.search(filename)
            if match:
                n = int(match.group(1))
                if n < n_lo:
                    pairs_lo.append((n, filename))
                elif n > n_hi:
                    pairs_hi.append((n, filename))
        pairs_lo.sort()
        pairs_hi.sort(reverse=True)
        if pairs_lo:
            num = len(pairs_lo)
            print(f"Reviewing {num} images where {self.name} < {n_lo} ({lo}%)")
            self.feh([filename for n, filename in pairs_lo])
        if pairs_hi:
            num = len(pairs_hi)
            print(f"Reviewing {num} images where {self.name} > {n_hi} ({hi}%)")
            self.feh([filename for n, filename in pairs_hi])


Histogram("width", r"_w(\d+)_").review(2, 97)
Histogram("height", r"_h(\d+)\.").review(2, 95)
Histogram("left", r"_l(\d+)_").review(2, 100)
Histogram("right", r"_r(\d+)_").review(0, 98)
