import os
from pathlib import Path

import cv2
import numpy as np

# Input videos are assumed to be organized as:
# videos/
#   fight/
#   normal/
#   theft/
#   threat/
VIDEOS_ROOT = Path("videos")
OUTPUT_ROOT = Path("data/frames")
FRAMES_PER_VIDEO = 50  # <-- 10 frames per video


def extract_n_frames_uniform(video_path: Path, out_dir: Path, n: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"[WARN] Cannot read frames from {video_path}")
        cap.release()
        return

    # Choose up to n indices uniformly across the video
    if total_frames <= n:
        idxs = list(range(total_frames))
    else:
        idxs = np.linspace(0, total_frames - 1, n, dtype=int)

    for i, frame_idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[WARN] Failed to read frame {frame_idx} from {video_path}")
            continue

        out_name = f"{video_path.stem}_f{i:02d}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), frame)

    cap.release()


def main():
    if not VIDEOS_ROOT.exists():
        raise SystemExit(f"{VIDEOS_ROOT} does not exist")

    for class_dir in VIDEOS_ROOT.iterdir():
        if not class_dir.is_dir():
            continue

        label = class_dir.name  # fight / normal / theft / threat
        out_class_dir = OUTPUT_ROOT / label

        print(f"[INFO] Processing class: {label}")

        for video_path in class_dir.glob("*.mp4"):
            print(f"  - {video_path.name}")
            extract_n_frames_uniform(video_path, out_class_dir, FRAMES_PER_VIDEO)


if __name__ == "__main__":
    main()
