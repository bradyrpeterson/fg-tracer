# fg-vision

A first computer vision project: a golf-style "shot tracer" for field goal kicks.
The goal is to track the football through field-goal videos and draw its flight
path. Python and OpenCV handle the video processing. Raw clips go in `data/raw/`,
and `data/clips.csv` records the details of each one.

## Setup

```bash
# create the virtual environment (first time only)
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy

# each new terminal session: activate the venv
source .venv/bin/activate

# when you're done
deactivate
```

## Usage

```bash
python src/inspect_video.py data/raw/kick_01.mp4   # inspect one clip
python src/inspect_all.py                          # inspect every clip in data/raw
```
