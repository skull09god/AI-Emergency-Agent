# detection/detector.py

# Stub motion detector for deployment without model files.

from typing import Literal, TypedDict


class DetectionResult(TypedDict):
    pose: str
    confidence: float


Label = Literal["normal", "fight", "theft", "threat"]


def simple_detector(
    motion_mean: float,
    motion_ratio: float,
    motion_std: float,
    motion_pixels: float,
    frame_idx: int,
) -> DetectionResult:
    """
    Dummy motion detector used in the deployed backend.
    All arguments are unused in this stub.
    """
    return {
        "pose": "normal",
        "confidence": 0.5,
    }
