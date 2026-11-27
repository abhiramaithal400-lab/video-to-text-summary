import os
import cv2
from config import ALLOWED_FORMATS, MAX_FILE_SIZE_MB, CLIP_LENGTH_MAX


def validate_video(file_path):
    # Check a: File extension is in ALLOWED_FORMATS
    ext = os.path.splitext(file_path)[1].lstrip(".").lower()
    if ext not in ALLOWED_FORMATS:
        return (False, f"Invalid format '{ext}'. Allowed formats: {ALLOWED_FORMATS}")

    # Check b: File size is at most MAX_FILE_SIZE_MB
    if not os.path.exists(file_path):
        return (False, f"File does not exist: {file_path}")

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return (False, f"File size {file_size_mb:.2f} MB exceeds limit of {MAX_FILE_SIZE_MB} MB")

    # Check c: File opens with cv2.VideoCapture
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return (False, "Failed to open video file with OpenCV")

    # Check d: Video has frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return (False, "Video contains no frames")

    # Check e: Duration is at most CLIP_LENGTH_MAX seconds
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0.0

    if duration > CLIP_LENGTH_MAX:
        cap.release()
        return (False, f"Video duration {duration:.1f}s exceeds maximum of {CLIP_LENGTH_MAX} seconds")

    cap.release()
    return (True, "")
