import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    EVENT_LABELS,
    FEATURE_DIM,
    FRAME_RATE,
    WINDOW_SIZE,
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
)
from detector import EventLSTM, extract_features, merge_nearby_events
from frame_sampler import sample_frames
from evaluation import precision_recall

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LABELS_FILE = DATA_DIR / "labels.csv"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"


def load_or_extract_features(clip_path):
    # Load cached ResNet features or extract and save them
    clip_name = os.path.basename(clip_path)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    feat_file = FEATURES_DIR / f"{clip_name}.npy"
    time_file = FEATURES_DIR / f"{clip_name}.times.npy"

    if feat_file.exists() and time_file.exists():
        return np.load(feat_file), np.load(time_file)

    print(f"Sampling frames and extracting features for {clip_name}...")
    sampled = sample_frames(str(clip_path), frame_rate=FRAME_RATE)
    if not sampled:
        return np.empty((0, FEATURE_DIM), dtype=np.float32), np.array([])

    timestamps = np.array([ts for ts, _ in sampled], dtype=np.float32)
    frames = [frame for _, frame in sampled]

    features = extract_features(frames)

    np.save(feat_file, features)
    np.save(time_file, timestamps)
    return features, timestamps


def build_training_windows():
    # Read labels and generate windowed dataset
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")

    df_labels = pd.read_csv(LABELS_FILE)
    window_len = max(1, WINDOW_SIZE * FRAME_RATE)

    all_windows = []
    all_targets = []
    clip_annotations = {}

    for _, row in df_labels.iterrows():
        clip = str(row["clip_name"]).strip()
        ts = float(row["time_seconds"])
        ev = str(row["event"]).strip().lower()
        if clip not in clip_annotations:
            clip_annotations[clip] = []
        clip_annotations[clip].append((ts, ev))

    for clip_name, events in clip_annotations.items():
        clip_path = DATA_DIR / clip_name
        if not clip_path.exists():
            print(f"Warning: Clip {clip_name} not found in {DATA_DIR}. Skipping.")
            continue

        features, times = load_or_extract_features(clip_path)
        if len(features) == 0:
            continue

        # Slide window across feature timeline
        for start in range(len(features)):
            end = min(start + window_len, len(features))
            window = features[start:end]

            # Pad window if near the end of video
            if len(window) < window_len:
                pad = np.repeat(window[-1:], window_len - len(window), axis=0)
                window = np.concatenate([window, pad], axis=0)

            start_t = float(times[start])
            end_t = start_t + WINDOW_SIZE

            # Check if any event falls in this window
            matched_label = "no_event"
            for ev_time, ev_label in events:
                if start_t <= ev_time <= end_t and ev_label in EVENT_LABELS:
                    matched_label = ev_label
                    break

            label_idx = EVENT_LABELS.index(matched_label)
            all_windows.append(window.astype(np.float32))
            all_targets.append(label_idx)

    if not all_windows:
        raise ValueError("No training windows could be generated.")

    X = np.stack(all_windows)
    y = np.array(all_targets, dtype=np.int64)
    return X, y, clip_annotations


def train_lstm(num_epochs=15, batch_size=16, learning_rate=0.001):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building feature windows from labeled clips...")
    X, y, clip_annotations = build_training_windows()
    print(f"Dataset ready: {len(X)} windows, {len(EVENT_LABELS)} classes.")

    # Class weights to handle class imbalance (more no_event than goals/cards)
    counts = np.bincount(y, minlength=len(EVENT_LABELS))
    weights = np.ones(len(EVENT_LABELS), dtype=np.float32)
    for i in range(len(EVENT_LABELS)):
        if counts[i] > 0:
            weights[i] = len(y) / (len(EVENT_LABELS) * counts[i])
        else:
            weights[i] = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = EventLSTM(
        input_dim=FEATURE_DIM,
        hidden_dim=128,
        num_classes=len(EVENT_LABELS),
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []

    print("Starting training loop...")
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch}/{num_epochs}] - Loss: {avg_loss:.4f} - Accuracy: {acc:.2%}")

    # Save trained model weights
    save_path = str(BASE_DIR / MODEL_PATH)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved trained model to {save_path}")

    # Plot training curve
    curve_file = MODELS_DIR / "training_curve.png"
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, num_epochs + 1), loss_history, marker="o", color="#2563EB", label="Training Loss")
    plt.title("EventLSTM Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_file, dpi=150)
    plt.close()
    print(f"Saved loss curve to {curve_file}")

    # Evaluate model predictions against labeled ground truth events
    model.eval()
    all_pred_events = []
    all_true_events = []

    with torch.no_grad():
        for clip_name, true_events in clip_annotations.items():
            clip_path = DATA_DIR / clip_name
            if not clip_path.exists():
                continue
            features, times = load_or_extract_features(clip_path)
            if len(features) < WINDOW_SIZE:
                continue

            for ev in true_events:
                all_true_events.append((ev[0], ev[1]))

            clip_raw_events = []
            for i in range(0, len(features) - WINDOW_SIZE + 1):
                window = features[i : i + WINDOW_SIZE]
                w_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(w_tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item())
                label = EVENT_LABELS[pred_idx]
                time_at_window = float(times[i + WINDOW_SIZE - 1])

                if label != "no_event" and confidence >= CONFIDENCE_THRESHOLD:
                    clip_raw_events.append((time_at_window, label, confidence))

            merged_clip_events = merge_nearby_events(clip_raw_events)
            all_pred_events.extend(merged_clip_events)

    precision, recall = precision_recall(all_pred_events, all_true_events, tolerance_seconds=10)
    print(f"Evaluation: Precision = {precision:.2f}, Recall = {recall:.2f}")

    metrics = {
        "precision": float(round(precision, 2)),
        "recall": float(round(recall, 2)),
        "epochs": num_epochs,
        "final_loss": float(round(loss_history[-1], 4)),
    }
    metrics_file = MODELS_DIR / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_file}")


if __name__ == "__main__":
    train_lstm(num_epochs=15)
