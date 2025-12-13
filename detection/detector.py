# detection/detector.py

# Stub motion detector for deployment without model files.
# Keeps the same function name so backend/app.py imports still work.

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

    Args:
        motion_mean: Mean motion (unused in stub).
        motion_ratio: Motion ratio (unused in stub).
        motion_std: Motion std (unused in stub).
        motion_pixels: Motion pixels (unused in stub).
        frame_idx: Frame index (unused in stub).

    Returns:
        Fixed label and confidence for demo.
    """
    # Always return a fixed label; adjust if you want a different default.
    return {
        "pose": "normal",
        "confidence": 0.5,
    }
