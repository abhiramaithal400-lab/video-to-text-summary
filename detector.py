import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, ResNetModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import (
    FEATURE_DIM,
    EVENT_LABELS,
    WINDOW_SIZE,
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
    MERGE_GAP_SECONDS,
)


def extract_features(frames):
    # Return empty array if no frames were provided
    if not frames:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    if not TORCH_AVAILABLE:
        print("Note: PyTorch / Transformers not installed. Using demo feature representations.")
        return np.zeros((len(frames), FEATURE_DIM), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained ResNet-50 image processor and model
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)
    model.eval()

    all_features = []
    batch_size = 16

    # Process frames in batches of 16
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i : i + batch_size]
        inputs = processor(images=batch_frames, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Take pooled output: [batch_size, 2048, 1, 1] -> [batch_size, 2048]
            pooled = outputs.pooler_output.squeeze(-1).squeeze(-1)
            all_features.append(pooled.cpu().numpy())

    return np.vstack(all_features)


if TORCH_AVAILABLE:
    class EventLSTM(nn.Module):
        # One LSTM layer, hidden size 128, dropout 0.3, and a linear output layer
        def __init__(self, input_dim=FEATURE_DIM, hidden_dim=128, num_classes=len(EVENT_LABELS), dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            # Input shape: [batch, window_len, FEATURE_DIM]
            lstm_out, _ = self.lstm(x)
            # Take the output from the final time step
            last_step = lstm_out[:, -1, :]
            out = self.dropout(last_step)
            logits = self.fc(out)
            return logits
else:
    class EventLSTM:
        def __init__(self, *args, **kwargs):
            pass


def detect_events(features, timestamps):
    # If PyTorch is unavailable or trained model file does not exist, run in DEMO MODE with clear warning
    if not TORCH_AVAILABLE or not os.path.exists(MODEL_PATH):
        if not TORCH_AVAILABLE:
            print("Note: PyTorch not installed in this environment. Running in DEMO MODE.")
        else:
            print(f"Warning: Model file not found at '{MODEL_PATH}'. Running in DEMO MODE.")
        demo_events = []
        if len(timestamps) > 0:
            max_time = timestamps[-1]
            if max_time >= 25:
                demo_events.append((23, "goal", 0.88))
            if max_time >= 45:
                demo_events.append((41, "card", 0.75))
            if not demo_events:
                demo_events.append((timestamps[len(timestamps) // 2], "goal", 0.82))
        return demo_events

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EventLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    num_frames = len(features)
    if num_frames == 0:
        return []

    detected = []

    # Slide window across features
    if num_frames < WINDOW_SIZE:
        pad_size = WINDOW_SIZE - num_frames
        pad = np.zeros((pad_size, FEATURE_DIM), dtype=np.float32)
        window_feats = np.vstack([features, pad])
        window_tensor = torch.tensor(window_feats, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(window_tensor)
            probs = F.softmax(logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
            label = EVENT_LABELS[pred_idx]

        if label != "no_event" and confidence >= CONFIDENCE_THRESHOLD:
            detected.append((timestamps[-1], label, confidence))
    else:
        for i in range(0, num_frames - WINDOW_SIZE + 1):
            window_feats = features[i : i + WINDOW_SIZE]
            window_tensor = torch.tensor(window_feats, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(window_tensor)
                probs = F.softmax(logits, dim=-1)[0]
                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item())
                label = EVENT_LABELS[pred_idx]

            timestamp = timestamps[i + WINDOW_SIZE - 1]
            if label != "no_event" and confidence >= CONFIDENCE_THRESHOLD:
                detected.append((timestamp, label, confidence))

    return detected


def merge_nearby_events(events):
    # Merge events with the same label closer than MERGE_GAP_SECONDS apart
    if not events:
        return []

    sorted_events = sorted(events, key=lambda x: x[0])
    merged = []

    for event in sorted_events:
        ts, label, conf = event
        if not merged:
            merged.append(event)
            continue

        last_ts, last_label, last_conf = merged[-1]
        if label == last_label and abs(ts - last_ts) < MERGE_GAP_SECONDS:
            if conf > last_conf:
                merged[-1] = event
        else:
            merged.append(event)

    return merged
