"""Build testdata/sample_faces.gif from faces_e5.jpg for CI selftest."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
photo = cv2.imread(str(root / "testdata" / "faces_e5.jpg"))
if photo is None:
    sys.exit("faces_e5.jpg missing")
h, w = photo.shape[:2]
side = min(h, w, 320)
x0, y0 = (w - side) // 2, (h - side) // 2
crop = photo[y0:y0 + side, x0:x0 + side]
frames = []
for i in range(6):
    dx, dy = (i % 3) - 1, ((i // 2) % 3) - 1
    M = np.float32([[1, 0, dx * 2], [0, 1, dy * 2]])
    fr = cv2.warpAffine(crop, M, (side, side), borderMode=cv2.BORDER_REFLECT)
    frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
out = root / "testdata" / "sample_faces.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0)
print(out, out.stat().st_size)
