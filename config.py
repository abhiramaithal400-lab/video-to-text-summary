# Configuration settings for video summary pipeline

FRAME_RATE = 1
FRAME_SIZE = 224
CLIP_LENGTH_MAX = 180          # seconds
WINDOW_SIZE = 10               # seconds of frames the LSTM looks at together
EVENT_LABELS = ["goal", "card", "foul", "corner", "no_event"]
CONFIDENCE_THRESHOLD = 0.6
MAX_FILE_SIZE_MB = 100
ALLOWED_FORMATS = ["mp4"]
MERGE_GAP_SECONDS = 5
MODEL_PATH = "models/lstm.pt"
FEATURE_DIM = 2048
