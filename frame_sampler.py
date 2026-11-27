import cv2
from config import FRAME_RATE, FRAME_SIZE


def load_video(path):
    # Open the video and read its frames-per-second
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, fps


def sample_frames(path, frame_rate=FRAME_RATE):
    # Sample frames at the specified rate and convert to RGB
    cap, fps = load_video(path)
    if not cap.isOpened():
        return []

    if fps <= 0:
        fps = 30.0

    frames = []
    frame_interval = fps / frame_rate
    frame_idx = 0
    next_sample_frame = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= next_sample_frame:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_seconds = round(frame_idx / fps, 2)
            frames.append((timestamp_seconds, rgb_frame))
            next_sample_frame += frame_interval

        frame_idx += 1

    cap.release()
    return frames


def resize_frame(frame, size=FRAME_SIZE):
    # Resize frame to size x size
    return cv2.resize(frame, (size, size))
