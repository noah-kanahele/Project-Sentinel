#!/usr/bin/env python3
"""Create or correct ground-truth labels for a clip.

  python scripts/label_clip.py data/clip.mov --out data/clip.labels.csv --auto color --config configs/gumball.yaml

--auto color pre-fills labels with the color detector (then review them); --auto none starts
with every frame "not visible". Review window keys (frames are shown downscaled; labels are
stored in full-resolution pixels):

  click        set the ball center on this frame (marks it visible)
  space / d    next frame            a    previous frame
  w / s        jump +10 / -10 frames
  x            mark NOT visible (ball out of frame or hidden)
  [ / ]        shrink / grow the labeled ball size by 10%
  o            save                  q    save and quit

Frame indices match scripts/run_video.py and scripts/evaluate_accuracy.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from edge_vision.camera.frame_source import VideoFileSource
from edge_vision.config import load_config
from edge_vision.detection.color import ColorBlobDetector
from edge_vision.evaluation.labels import Label, autolabel, load_labels, save_labels

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
WINDOW = "label clip"
MAX_SHOW_WIDTH = 960


def load_frames(path: str) -> List[np.ndarray]:
    frames = []
    with VideoFileSource(path) as src:
        for f in src:
            frames.append(f.image)
    return frames


def review(frames: List[np.ndarray], labels: Dict[int, Label], out: str) -> None:
    h, w = frames[0].shape[:2]
    scale = min(1.0, MAX_SHOW_WIDTH / w)
    state = {"i": 0, "click": None}

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x / scale, y / scale)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    last_size = next((lb.size for lb in labels.values() if lb.size), 40.0)
    while True:
        i = state["i"]
        label = labels[i]
        if state["click"] is not None:
            size = label.size or last_size
            label = labels[i] = Label(i, True, state["click"][0], state["click"][1], size, size)
            state["click"] = None
        if label.size:
            last_size = label.size
        img = cv2.resize(frames[i], (int(w * scale), int(h * scale)))
        if label.visible and label.cx is not None:
            c = (int(label.cx * scale), int(label.cy * scale))
            r = int((label.size or last_size) * scale / 2)
            cv2.circle(img, c, r, (0, 255, 0), 2)
            cv2.drawMarker(img, c, (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
        visible = sum(lb.visible for lb in labels.values())
        status = "VISIBLE" if label.visible else "NOT VISIBLE"
        color = (0, 255, 0) if label.visible else (0, 0, 255)
        cv2.putText(img, f"frame {i}/{len(frames) - 1}  {status}  ({visible} visible total)", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(img, "click=set  x=hidden  space/a=next/prev  w/s=+-10  [ ]=size  o=save  q=quit", (12, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        ch = chr(key)
        if ch in (" ", "d"):
            state["i"] = min(i + 1, len(frames) - 1)
        elif ch == "a":
            state["i"] = max(i - 1, 0)
        elif ch == "w":
            state["i"] = min(i + 10, len(frames) - 1)
        elif ch == "s":
            state["i"] = max(i - 10, 0)
        elif ch == "x":
            labels[i] = Label(i, False)
        elif ch in "[]" and label.visible:
            f = 0.9 if ch == "[" else 1.1
            labels[i] = Label(i, True, label.cx, label.cy, (label.w or last_size) * f, (label.h or last_size) * f)
        elif ch == "o":
            save_labels(out, labels.values())
            print(f"saved {out}")
        elif ch == "q":
            break
    save_labels(out, labels.values())
    cv2.destroyWindow(WINDOW)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", required=True, help="labels CSV to write (also read if it exists)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--auto", choices=["color", "none"], default="color", help="how to pre-fill labels")
    ap.add_argument("--no-review", action="store_true", help="only write the pre-filled labels, no window")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if Path(args.out).exists():
        labels = load_labels(args.out)
        print(f"loaded {len(labels)} existing labels from {args.out}")
    elif args.auto == "color":
        with VideoFileSource(args.video) as src:
            labels = {lb.frame_idx: lb for lb in autolabel(src, ColorBlobDetector.from_config(cfg))}
        print(f"auto-labeled {len(labels)} frames, {sum(lb.visible for lb in labels.values())} visible")
    else:
        with VideoFileSource(args.video) as src:
            labels = {f.index: Label(f.index, False) for f in src}

    if args.no_review:
        save_labels(args.out, labels.values())
        print(f"wrote {args.out}")
        return
    review(load_frames(args.video), labels, args.out)


if __name__ == "__main__":
    main()
