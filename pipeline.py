import sys
from validation import validate_video
from frame_sampler import sample_frames
from detector import extract_features, detect_events, merge_nearby_events
from text_generator import build_summary
from database import record_video_analysis, init_db


def summarize_video(video_path):
    # Step 1: Validate the video
    is_valid, error_message = validate_video(video_path)
    if not is_valid:
        raise ValueError(error_message)

    # Step 2: Sample frames from the video
    samples = sample_frames(video_path)
    if not samples:
        return build_summary([]), []

    timestamps = [ts for ts, _ in samples]
    frames = [frame for _, frame in samples]

    # Step 3: Extract deep features from frames
    features = extract_features(frames)

    # Step 4: Detect key match events
    detected = detect_events(features, timestamps)

    # Step 5: Merge events occurring close to each other
    merged = merge_nearby_events(detected)

    # Step 6: Generate textual summary
    summary_text = build_summary(merged)

    # Step 7: Record analysis results in SQLite database
    try:
        duration = float(timestamps[-1]) if timestamps else 0.0
        record_video_analysis(video_path, duration, merged, summary_text)
    except Exception as db_err:
        print(f"[Database Log Warning]: {db_err}")

    return summary_text, merged


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        try:
            summary, events = summarize_video(video_file)
            print("Summary:")
            print(summary)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python pipeline.py <path_to_video.mp4>")
