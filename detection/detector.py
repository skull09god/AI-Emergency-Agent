import numpy as np
import joblib
import os

# Load the trained model and label names
MODEL_PATH = os.path.join(os.path.dirname(__file__), "motion_model.joblib")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
LABEL_NAMES = bundle["label_names"]  # e.g. {0: "normal", 1: "fight", 2: "theft", 3: "threat"}


def simple_detector(motion_mean, motion_ratio, motion_std, motion_pixels, frame_idx):
    """
    Predict class (normal / fight / theft / threat) from motion features.
    These 5 features must match train_model.py.
    """
    features = np.array([[motion_mean, motion_ratio, motion_std, motion_pixels, frame_idx]])
    probs = model.predict_proba(features)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    label = LABEL_NAMES[pred_class]

    return {
        "pose": label,
        "confidence": confidence,
    }


if __name__ == "__main__":
    # quick test on dummy features
    result = simple_detector(5.0, 0.01, 10.0, 1000, 10)
    print(result)
