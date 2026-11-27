import evaluate


def precision_recall(predicted_events, true_events, tolerance_seconds=5):
    # If both lists are empty, precision and recall are perfect
    if not predicted_events and not true_events:
        return (1.0, 1.0)
    if not predicted_events or not true_events:
        return (0.0, 0.0)

    matched_true_indices = set()
    true_positives = 0

    # Match each prediction with an unmatched true event within tolerance
    for pred in predicted_events:
        pred_ts, pred_label = pred[0], pred[1]

        for idx, true_ev in enumerate(true_events):
            if idx in matched_true_indices:
                continue

            true_ts, true_label = true_ev[0], true_ev[1]
            if pred_label == true_label and abs(pred_ts - true_ts) <= tolerance_seconds:
                matched_true_indices.add(idx)
                true_positives += 1
                break

    precision = true_positives / len(predicted_events)
    recall = true_positives / len(true_events)
    return (precision, recall)


def rouge_score(generated_summary, reference_summary):
    # Compute ROUGE score using Hugging Face evaluate
    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=[generated_summary],
        references=[reference_summary]
    )
    return scores
