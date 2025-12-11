import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Map folder names to numeric labels
# folders: videos/normal, videos/fight, videos/theft, videos/threat
CLASS_MAP = {
    "normal": 0,
    "fight": 1,
    "theft": 2,   # theft + break-ins
    "threat": 3,
}

LABEL_NAMES = {v: k for k, v in CLASS_MAP.items()}

VIDEOS_ROOT = "videos"


def extract_motion_features(video_path, max_frames=300):
    """
    Extract simple motion-based features from a video file.
    Returns an array of shape (n_samples, n_features).
    """
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    features = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            h, w = thresh.shape
            total_pixels = h * w

            motion_sum = float(np.sum(thresh))
            motion_mean = motion_sum / total_pixels
            motion_pixels = int(np.sum(thresh > 0))
            motion_ratio = motion_pixels / total_pixels
            motion_std = float(np.std(thresh[thresh > 0])) if motion_pixels > 0 else 0.0

            # feature vector for this frame pair
            features.append([
                motion_mean,
                motion_ratio,
                motion_std,
                motion_pixels,
                frame_idx,
            ])

        prev_gray = gray
        frame_idx += 1
        if frame_idx >= max_frames:
            break

    cap.release()
    return np.array(features)


def load_dataset():
    """
    Walk through videos/<class_name> folders, extract features and labels.
    """
    X_list = []
    y_list = []

    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(VIDEOS_ROOT, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: folder not found: {class_dir}")
            continue

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            video_path = os.path.join(class_dir, fname)
            print(f"Processing {video_path} as {class_name} ({label})")

            feats = extract_motion_features(video_path)
            if feats.size == 0:
                print(f"  No features extracted from {video_path}")
                continue

            X_list.append(feats)
            y_list.append(np.full(len(feats), label, dtype=int))

    if not X_list:
        raise RuntimeError("No training data found. Check your videos folders.")

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    print(f"Total samples: {X.shape[0]}, features per sample: {X.shape[1]}")
    return X, y


def main():
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"Validation accuracy: {acc:.3f}")

    os.makedirs("detection", exist_ok=True)
    model_path = os.path.join("detection", "motion_model.joblib")
    joblib.dump({"model": clf, "label_names": LABEL_NAMES}, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
