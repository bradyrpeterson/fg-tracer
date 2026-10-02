# fg-vision

A golf-style "shot tracer" for field goal kicks. Give it a broadcast clip of a
field goal (camera behind the kicker) and it draws the ball's flight from the
holder's hands through the uprights. Python, OpenCV and a small YOLO model
trained on our own clips do the work. Raw clips go in `data/raw/`, and
`data/clips.csv` records the details of each one.

## Quick start: trace your own video

```bash
git clone https://github.com/bradyrpeterson/fg-vision.git
cd fg-vision
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # first time only (a few minutes)

python src/trace.py path/to/your_kick.mp4                  # full tracer line
python src/trace.py path/to/your_kick.mp4 --style comet    # or a comet on the ball
```

The result is saved in `outputs/` (`your_kick_tracer.mp4` or `your_kick_comet.mp4`).
The trained ball detector ships with the repo (`models/ball/weights/best.pt`),
so nothing else needs downloading or training.

**Best results:** a short clip of one kick (a few seconds before the snap until
the ball passes the uprights), broadcast camera behind the kicker, original
quality rather than a screen recording. Options: `--start kick` starts the line
at the holder instead of once the ball is in the air; `python src/trace.py --help`
lists everything.

Runs on Mac, Windows or Linux; it uses an Apple or NVIDIA GPU if there is one,
otherwise the CPU (slower, but it works).

## How it works

| Step | File | What it does |
|---|---|---|
| 1 | `motion.py` | Reads frames (skipping duplicates), measures how the camera moved between frames, finds screen graphics to ignore |
| 2 | `kick.py` | Finds when the camera starts following the kick and where the holder is |
| 3 | `field.py` | One "field" coordinate system for the whole clip, so positions stay stuck to the field while the camera moves |
| 4 | `stack.py` | Turns three neighboring frames into one image where moving things show up as red-green-blue fringes |
| 5 | `detect_ball.py` | Runs the trained YOLO detector on those motion images, in full-resolution tiles |
| 6 | `trajectory.py` | Links detections into the ball's track, then fits the physics of a kick (a parabola seen through a camera), so the line can't jab or wiggle; a slow drift correction keeps it on the ball; stray sightings and stragglers after the ball is gone are dropped |
| 7 | `draw.py` | Draws a smooth, glowing tracer line and saves high-quality H.264 video |

`track_ball.py` is the older tracker that needs no trained model. It's used to
make the very first training labels.

## Teaching the detector (and adding new videos)

The detector learns from our own clips, labeled automatically ("pseudo-labels"):

```bash
# 1. make training tiles (first time: no model, uses track_ball.py)
python src/make_dataset.py data/raw/kick_0*.mp4
# later rounds: use the current detector; also saves its mistakes as examples
python src/make_dataset.py data/raw/kick_0*.mp4 --model models/ball/weights/best.pt

# 1b. synthetic examples: a drawn football flying over any real football footage
python src/synth.py --count 4000 data/raw/kick_0*.mp4 data/external/*.webm

# 2. train in the background (keeps going if the terminal closes; Mac stays awake)
#    the --test clips are never trained on, so their score is honest
./train_long.sh ball 25 --train kick_01 kick_02 kick_03 kick_04 kick_06 synthetic --test kick_05
tail -f models/ball.log

# 3. measure how close the line is to the real ball
python src/evaluate.py kick_05
```

To add a new video: put it in `data/raw/`, run `trace.py` on it, and watch the
result. If the line is right, include it in the next `make_dataset.py` run and
retrain. Every clip added this way makes the detector better on the next new one.

`data/external/` holds openly licensed football videos from Wikimedia Commons
(see `data/external/CREDITS.md`), used as synthetic-data backgrounds;
`nw_pat.mp4` (an extra point from a FOX highlight reel) is a test clip from a
stadium the detector never trained on.

`labels/ground_truth.json` holds ball positions checked by eye; `evaluate.py`
uses them to score the tracer.

## Other tools

```bash
python src/inspect_video.py data/raw/kick_01.mp4   # resolution, fps, duration, first frame
python src/inspect_all.py                          # the same for every clip in data/raw
python src/motion_preview.py data/raw/kick_01.mp4  # where is there motion? (raw frames)
python src/track_ball.py data/raw/kick_01.mp4      # the older, model-free tracker
```
