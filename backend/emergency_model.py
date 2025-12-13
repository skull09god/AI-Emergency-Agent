# backend/emergency_model.py

# Stub implementation used for deployment without heavy ML dependencies.

from typing import Literal

Label = Literal["emergency", "non_emergency"]


def classify_image(image_path: str) -> Label:
    """
    Dummy classifier used in the deployed backend.

    Args:
        image_path: Path to the saved image file.

    Returns:
        A simple fixed label. Replace with real model logic later.
    """
    return "emergency"
