"""Check the green / yellow / red verdict against what really happened.

For each clip: the final verdict vs actual_result in data/clips.csv, and how
long after the kick the verdict became right (and stayed right). Clips with no
result yet, or SHORT / BLOCKED (only left / right is judged), are skipped.

    python src/check_verdicts.py kick_01 kick_02 kick_03 kick_04 kick_05 kick_06
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from trace import playback_fps, trace
from verdict import kick_distance, label, verdicts


def check(clip, model, actual):
    r = trace(clip, model)
    fps = playback_fps(clip, len(r["frames"]))
    chance = verdicts(r, fps, kick_distance(clip))
    want = "GOOD" if actual == "MADE" else "NO GOOD"
    labels = [label(p) for p in chance]
    final = labels[r["end"]]
    # first frame after which the verdict is right all the way to the end
    right_from = next((i for i in range(len(labels) - 1, -1, -1) if labels[i] != want), -1) + 1
    when = f"right from {(right_from - r['kick']) / fps:.1f}s after the kick" if right_from <= r["end"] else "never right"
    flight = (r["end"] - r["kick"]) / fps
    print(f"  {clip.stem}: {actual:10s} -> {final:8s} (chance {chance[r['end']]:.2f}); {when} "
          f"(ball tracked for {flight:.1f}s)")
    return final == want


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--model", default="models/ball/weights/best.pt")
    args = parser.parse_args()
    results = {row["clip_id"]: row["actual_result"] for row in csv.DictReader(open("data/clips.csv"))}
    model = YOLO(args.model)
    scored = [c for c in args.clips if results.get(c) in ("MADE", "MISS LEFT", "MISS RIGHT")]
    for c in sorted(set(args.clips) - set(scored)):
        print(f"  {c}: skipped (actual_result is '{results.get(c, '')}')")
    right = [check(Path(f"data/raw/{c}.mp4"), model, results[c]) for c in scored]
    print(f"final verdict right for {sum(right)} of {len(right)} clips")
