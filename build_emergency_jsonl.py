import os
import json

FRAME_ROOT = "data/frames"
OUTPUT_JSONL = "data/emergency_vl.jsonl"

CLASSES = ["normal", "fight", "theft", "threat"]

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
    for cls in CLASSES:
        frame_dir = os.path.join(FRAME_ROOT, cls)
        for fname in os.listdir(frame_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(FRAME_ROOT, cls, fname).replace("\\", "/")

            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Classify the emergency type in this frame as one of: "
                            "normal, fight, theft, threat.\n"
                            f"Image path: {image_path}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": cls,
                    },
                ]
            }

            f_out.write(json.dumps(example) + "\n")
            print(f"Wrote example for {image_path}")
