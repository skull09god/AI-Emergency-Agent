import cv2
import numpy as np
from detection.detector import simple_detector

def run_on_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    print("Opened:", cap.isOpened())

    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return

    cv2.namedWindow("ERSA - Video Motion Detector", cv2.WINDOW_NORMAL)
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("End of video, exiting")
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

            result = simple_detector(
                motion_mean=motion_mean,
                motion_ratio=motion_ratio,
                motion_std=motion_std,
                motion_pixels=motion_pixels,
                frame_idx=frame_idx,
            )
            label = f"{result['pose']} ({result['confidence']:.2f})"

            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        prev_gray = gray
        frame_idx += 1

        cv2.imshow("ERSA - Video Motion Detector", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Stopped by user")
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    run_on_video(
        r"C:\Users\prabh\OneDrive\Desktop\AI-Emergency-Agent\videos\threat\threat_03.mp4"
    )
