# fg-vision

A golf-style shot tracer for field goal kicks: give it a broadcast clip (camera
behind the kicker) and it draws the ball's flight from the holder through the
uprights. See README.md for setup and the full pipeline.

## Working with the user
- The user is learning computer vision as they go. Explain each step in plain
  language before doing it, and say how to check it worked afterwards.
- Keep code short and readable, with brief comments; match the existing style.
- Ask before big direction changes; report results honestly (including what didn't work).

## Commands
```bash
source .venv/bin/activate
python src/trace.py data/raw/kick_05.mp4        # trace a clip -> outputs/<clip>_tracer.mp4
python src/evaluate.py kick_01 kick_03 kick_05  # line accuracy vs labels/ground_truth.json
python src/label.py data/raw/kick_09.mp4        # click the ball by hand -> labels/hand_labels.json
./train_long.sh <name> <epochs> --train ... --test kick_05   # detached training (survives time limits)
```

## Things to know
- `outputs/*_tracer.mp4` is the current tracer (trace.py). By default the line starts
  ~1/4 s after the kick (`--start airborne`, `AIRBORNE_FRAMES` in trace.py) and fades in;
  `--start kick` starts at the holder, `--style comet` draws a short tail on the ball.
- Long jobs: tool background commands are killed after ~2 hours. Run training with
  `train_long.sh` (nohup + caffeinate) and poll `models/<name>/results.csv`.
- `data/` and `outputs/` are gitignored. In `models/`, only the shipped detector
  `models/ball/weights/best.pt` (trained on all 6 clips + synthetic data) is committed,
  so a fresh clone can trace videos right away; other models stay local. When a
  retrained model is better, replace that file and commit it.
- Detector results are cached per model in `outputs/cache/`.
- Honest testing: the current model trained on all 6 clips, so only
  `data/external/nw_pat.mp4` (new stadium) is still unseen. When adding clips, hold
  some out of training to measure generalization. Ground truth in `labels/ground_truth.json` came from
  the old tracker, so it's only accurate to a few pixels.
- Broadcast NFL clips are copyrighted: don't bulk-download them. Open footage
  (Wikimedia Commons, see `data/external/CREDITS.md`) and synthetic data (`src/synth.py`) are fine.

## Roadmap: track (almost) any kick
Goal: work on wide/still cameras and college games, with the ball itself (not the
camera tilting) deciding when the kick happens.
1. Hand-label 20-30 varied clips with `label.py` (DONE: tool + `make_dataset.py --labels`
   + `evaluate.py --truth`). Hold ~1/3 out of training.
2. Kick timing from the ball: a track that starts at the holder and rises (kick.py);
   camera tilt only as a hint. Hand labels' "kick" frame checks it.
3. Retrain on hand labels + synthetic (synth.py with the new stadiums), several image sizes.
4. Measure on held-out clips (evaluate.py, check_verdicts.py).

## Roadmap: color-coded tracer (FIRST VERSION DONE: `--verdict`)
Status: `uprights.py` finds yellow posts/bar; `verdict.py` refits the curve on past
sightings each frame and reads the ball's position at the goal over a crossing-time
window (distance / 15-20 yd/s, because depth can't be measured from behind: a 3D
gravity fit was tried and failed). `check_verdicts.py`: 6/6 final verdicts right,
but the window and thresholds were tuned on those same clips. Next: more labeled
clips (especially misses), white college posts, reading "NN YD ATTEMPT" from the screen.
Crossbar / short kicks: removed for now (the user wants it back later). Judging it
from ball height on screen failed on long low kicks (kick_07, kick_08 went "unsure").
Ideas: officials' signal (after the fact only), ball size (too few pixels), side angle.

Original plan:
Goal: like a golf shot tracer, the line is **green** when the kick looks good,
**yellow** when unsure, **red** when it's missing / off target.

1. **Detect the uprights.** Bright yellow, "Y"-shaped: find both posts and the
   crossbar by color and shape, and place them in field coordinates (`field.py`).
2. **"Past-only" prediction mode.** The current physics fit (`trajectory.fit_flight`)
   uses the whole flight, so it "knows the future". For honest colors, the
   prediction at frame i may only use sightings up to frame i. Keep the
   whole-flight fit for drawing the smooth line.
3. **Predict the crossing point + confidence.** From the physics curve, predict
   where the ball passes the goal relative to the posts and bar, with an
   uncertainty that shrinks as more of the flight is seen.
   - Left/right: reliable from behind the kicker.
   - Short (height at the goal line): hard in 2D; use the curve's depth term plus
     the kick distance (`data/clips.csv` distance_yds, or the on-screen "NN YD ATTEMPT").
4. **Color the line.** Green = confidently inside the posts and above the bar;
   yellow = early / close to a post or the bar; red = confidently wide or short.
5. **Check against reality.** `data/clips.csv` has actual_result (MADE / MISS LEFT /
   MISS RIGHT / SHORT). Six clips can't show accuracy; aim for 30+ with plenty of
   misses. Measure: how often the final color is right, and how early it becomes right.

Suggested order: 1 -> 2 (validated against clips.csv) -> 3 -> 4.
