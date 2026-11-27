def format_event_line(timestamp, label):
    # Format minute number with a prime symbol
    minute = int(round(timestamp))

    # "card" becomes "Yellow card", other labels are capitalised
    if label.lower() == "card":
        display_label = "Yellow card"
    else:
        display_label = label.capitalize()

    return f"{minute}' - {display_label}"


def build_summary(events):
    # Return message if no events were detected
    if not events:
        return "No key events detected."

    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda x: x[0])

    # Format one line per event
    lines = [format_event_line(ev[0], ev[1]) for ev in sorted_events]

    return "\n".join(lines)
