from zoneinfo import ZoneInfo  # Python 3.9+
from .emergency_model import classify_image
from pathlib import Path
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import shutil
import sys
import cv2
import numpy as np
import logging
from collections import Counter
import warnings
from datetime import datetime, timezone
from typing import List, Dict, Any
import httpx


# ---------------- Telegram config ----------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


async def send_telegram_message(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured, skipping notification")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload)
        print("Telegram status:", resp.status_code)
        try:
            print("Telegram response:", resp.json())
        except Exception:
            print("Telegram raw response:", resp.text)

# -------------------------------------------------

# Make sure we can import detection/
sys.path.append("..")
from detection.detector import simple_detector  # Correct function name

warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI(title="Emergency Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# In-memory incident log
INCIDENTS: List[Dict[str, Any]] = []
INCIDENT_COUNTER = 0


# NEW: simple image classification endpoint using your ResNet model
@app.post("/classify-frame")
async def classify_frame(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    tmp_dir = Path("tmp_frames")
    tmp_dir.mkdir(exist_ok=True)
    suffix = Path(file.filename).suffix or ".jpg"
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    contents = await file.read()
    tmp_path.write_bytes(contents)

    label = classify_image(str(tmp_path))

    tmp_path.unlink(missing_ok=True)

    return {"label": label}


@app.post("/infer-video-resnet")
async def infer_video_resnet(video: UploadFile = File(...)):
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise HTTPException(status_code=400, detail="Invalid video")

        # Sample every 10th frame (adjust as needed)
        sample_every = max(1, total_frames // 50)  # ~50 frames max
        print(f"[INFO] Analyzing {total_frames} frames, sampling every {sample_every}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Classify every Nth frame with ResNet
            if frame_idx % sample_every == 0:
                tmp_dir = Path("tmp_frames")
                tmp_dir.mkdir(exist_ok=True)
                frame_path = tmp_dir / f"{uuid.uuid4().hex}.jpg"
                cv2.imwrite(str(frame_path), frame)

                label = classify_image(str(frame_path))
                predictions.append(label)

                frame_path.unlink(missing_ok=True)

            frame_idx += 1

        if not predictions:
            return {
                "event": "normal",
                "confidence": 0.0,
                "frames_analyzed": 0,
                "resnet_frames": 0,
            }

        # Majority vote
        majority_event = Counter(predictions).most_common(1)[0][0]
        conf = Counter(predictions).most_common(1)[0][1] / len(predictions)

        return {
            "event": majority_event,
            "confidence": float(conf),
            "frames_analyzed": total_frames,
            "resnet_frames": len(predictions),
        }

    finally:
        if cap is not None:
            cap.release()
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                pass


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
async def infer_incident(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    """
    Wraps infer_video_resnet and adds incident status based on rules:
    - high_risk if event != "normal" and confidence >= 0.8
    - low_risk otherwise
    Also logs each incident in memory and sends Telegram for high_risk.
    """
    global INCIDENT_COUNTER

    # use ResNet-based video inference
    result = await infer_video_resnet(video)

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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": video.filename,
        "event": event,
        "confidence": confidence,
        "frames_analyzed": response["frames_analyzed"],
        "status": status,
    }
    INCIDENTS.append(incident_entry)

    # convert UTC timestamp string to IST for display
    utc_dt = datetime.fromisoformat(incident_entry["timestamp"])
    ist_dt = utc_dt.astimezone(ZoneInfo("Asia/Kolkata"))
    ist_str = ist_dt.strftime("%Y-%m-%d %H:%M:%S")

    # send Telegram only for high-risk incidents
    if status == "high_risk":
        msg = (
            f"🚨 *High-risk incident detected!*\n"
            f"Event: {event}\n"
            f"Confidence: {confidence:.2f}\n"
            f"File: {video.filename}\n"
            f"Time (IST): {ist_str}"
        )
        background_tasks.add_task(send_telegram_message, msg)

    return response


@app.get("/incidents")
def list_incidents(limit: int = 50):
    """
    Return recent incidents (newest first), up to ?limit.
    """
    recent = list(reversed(INCIDENTS))
    return recent[:limit]


@app.post("/escalate")
def escalate(incident: Dict[str, Any]):
    """
    Simple escalation policy (rule-based for now).
    Expected body: { "event": "...", "confidence": 0.9, "status": "high_risk" }
    """
    status = incident.get("status", "low_risk")
    if status == "high_risk":
        decision = "ALERT_POLICE"
    else:
        decision = "MONITOR"
    return {"decision": decision, "oumi_used": False}


@app.get("/")
def root():
    return {"message": "Emergency Detection API ready! POST to /infer-video"}
