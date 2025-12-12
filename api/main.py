from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import shutil
import sys
import cv2
import numpy as np
import logging
from collections import Counter
import warnings
from datetime import datetime
from typing import List, Dict, Any

# Make sure we can import detection/
sys.path.append('..')
from detection.detector import simple_detector  # Correct function name

# Oumi RL
from oumi import Oumi

warnings.filterwarnings('ignore', category=UserWarning)

app = FastAPI(title='Emergency Detection API')
logging.basicConfig(level=logging.INFO)

# In-memory incident log
INCIDENTS: List[Dict[str, Any]] = []
INCIDENT_COUNTER = 0

# Try to load Oumi RL policy (optional at startup)
try:
    escalation_policy = Oumi.load("escalation_policy")
    print("✅ Loaded Oumi RL policy from escalation_policy/")
except Exception as e:
    escalation_policy = None
    print("⚠️ Oumi RL policy not loaded:", e)


async def run_infer_video(video: UploadFile):
    """
    Helper to reuse infer_video logic from other routes.
    """
    return await infer_video(video)


@app.post("/infer-video")
async def infer_video(video: UploadFile = File(...)):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name

    cap = None
    try:
        predictions = []
        cap = cv2.VideoCapture(tmp_path)
        ret, prev_frame = cap.read()
        if not ret:
            raise HTTPException(status_code=400, detail="Invalid video")

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        # detect initial points to track
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=5
        )
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_idx += 1

            # refresh points if lost
            if prev_pts is None or len(prev_pts) == 0:
                prev_pts = cv2.goodFeaturesToTrack(
                    prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=5
                )

            if prev_pts is not None and len(prev_pts) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_pts, None
                )
                good_new = next_pts[status == 1]
                good_old = prev_pts[status == 1]

                if len(good_new) > 0:
                    flow = good_new - good_old
                    mag = np.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)

                    # 5 features matching train_model.py idea
                    motion_mean = float(np.mean(mag))
                    motion_pixels = float(len(mag))
                    motion_std = float(np.std(mag))
                    motion_ratio = float(
                        np.sum(mag > 1.0) / motion_pixels
                    ) if motion_pixels > 0 else 0.0

                    result = simple_detector(
                        motion_mean, motion_ratio, motion_std, motion_pixels, frame_idx
                    )
                    predictions.append((result["pose"], result["confidence"]))

                # update points for next frame
                prev_pts = good_new.reshape(-1, 1, 2)

            prev_gray = gray

        if not predictions:
            return {
                "event": "normal",
                "confidence": 0.0,
                "frames_analyzed": 0,
            }

        events = [p[0] for p in predictions]
        confs = [p[1] for p in predictions]
        majority_event = Counter(events).most_common(1)[0][0]
        avg_conf = sum(confs) / len(confs)

        return {
            "event": majority_event,
            "confidence": float(avg_conf),
            "frames_analyzed": len(predictions),
        }

    finally:
        if cap is not None:
            cap.release()
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                # Windows may still hold a lock briefly; ignore if so
                pass


@app.post("/infer-incident")
async def infer_incident(video: UploadFile = File(...)):
    """
    Wraps infer_video and adds incident status based on rules:
    - high_risk if event != "normal" and confidence >= 0.8
    - low_risk otherwise
    Also logs each incident in memory.
    """
    global INCIDENT_COUNTER

    # run underlying video inference
    result = await run_infer_video(video)

    event = result["event"]
    confidence = float(result["confidence"])

    if event != "normal" and confidence >= 0.8:
        status = "high_risk"
    else:
        status = "low_risk"

    response = {
        "event": event,
        "confidence": confidence,
        "frames_analyzed": result.get("frames_analyzed", 0),
        "status": status,
    }

    # log incident in memory
    INCIDENT_COUNTER += 1
    incident_entry = {
        "id": INCIDENT_COUNTER,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "filename": video.filename,
        "event": event,
        "confidence": confidence,
        "frames_analyzed": response["frames_analyzed"],
        "status": status,
    }
    INCIDENTS.append(incident_entry)

    return response
