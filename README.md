# Sports Match Video to Text Summary (Football)

A Python 3.12 pipeline that converts a short football match video clip (MP4, 1 to 3 minutes) into a concise, timeline-based text summary of key events (such as goals, cards, and fouls).

## Pipeline Architecture

1. **Validation**: Checks file format, size, frame count, and maximum duration.
2. **Frame Sampler**: Samples frames at 1 frame per second and converts BGR to RGB.
3. **Detector**: Extracts ResNet-50 visual features, feeds sliding window sequences into a custom PyTorch LSTM classifier to detect match events, and merges nearby duplicate events. Includes a graceful demo fallback if no trained model checkpoint is present.
4. **Text Generator**: Formats detected timestamps into match minutes and capitalised labels (e.g., `23' - Goal`, `41' - Yellow card`).
5. **Evaluation**: Computes precision, recall (with time tolerance), and ROUGE summary quality scores.

## Project Structure

```text
video-summary/
├── app.py              # Streamlit interactive web interface
├── config.py           # Global settings and configuration variables
├── validation.py       # Video file validation checks
├── frame_sampler.py    # Frame extraction and resizing
├── detector.py         # ResNet-50 feature extraction & EventLSTM classifier
├── text_generator.py   # Event formatting and match summary generator
├── evaluation.py       # Precision, recall, and ROUGE scoring
├── pipeline.py         # End-to-end execution pipeline
├── train.py            # EventLSTM training and evaluation script
├── requirements.txt    # Project dependencies
├── README.md           # Documentation and usage guide
├── data/               # Input clips, labels.csv, and cached features
│   ├── labels.csv      # Clip timestamps and event annotations
│   └── features/       # Cached ResNet-50 feature vectors
└── models/             # PyTorch model weights and evaluation curves
    ├── lstm.pt         # Trained model weights
    ├── metrics.json    # Precision and recall metrics
    └── training_curve.png # Training loss curve plot
```

## Installation

Install the required dependencies using pip:

```bash
python -m pip install -r requirements.txt
```

## Training the Model

Train the EventLSTM model on the clips in `data/`:

```bash
python train.py
```

This extracts ResNet-50 features, saves the trained weights to `models/lstm.pt`, generates `models/training_curve.png`, and saves `models/metrics.json`.

## Running the Web App

Launch the interactive Streamlit website:

```bash
python -m streamlit run app.py
```

## Running the Pipeline via CLI

Run the pipeline directly on any football clip:

```bash
python pipeline.py path/to/match_clip.mp4
```

## Compilation & Verification

To compile and verify syntax for any module:

```bash
python -m py_compile train.py
```

Or verify all files across the project:

```bash
python -m py_compile config.py validation.py frame_sampler.py detector.py text_generator.py evaluation.py pipeline.py train.py app.py
```
