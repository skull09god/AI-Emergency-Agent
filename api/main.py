from fastapi import FastAPI
from pydantic import BaseModel

from detection.detector import simple_detector

app = FastAPI()


class MotionInput(BaseModel):
    motion_level: float


@app.post("/detect")
def detect(input: MotionInput):
    result = simple_detector(input.motion_level)
    return result
