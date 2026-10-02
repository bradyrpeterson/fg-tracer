"""Train a YOLO ball detector on the three-frame motion tiles from make_dataset.py.

    python src/train_detector.py --train kick_01 kick_03 kick_04 kick_06 --test kick_05

--test clips are never trained on, so their score shows how the detector
does on a video it has never seen (the real goal).
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from detect_ball import DEVICE  # Apple GPU, NVIDIA GPU or CPU, whichever this computer has

DATA = Path("data/ball_dataset").resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--name", default="ball")
    args = parser.parse_args()

    # YOLO reads a small YAML file that says where the images are
    yaml = Path("data/ball_dataset") / f"{args.name}.yaml"
    yaml.write_text(
        "train:\n" + "".join(f"  - {DATA / c / 'images'}\n" for c in args.train)
        + "val:\n" + "".join(f"  - {DATA / c / 'images'}\n" for c in args.test)
        + "names:\n  0: ball\n")

    model = YOLO(args.model)  # start from weights trained on everyday photos (COCO)
    model.train(
        data=str(yaml), epochs=args.epochs, imgsz=640, batch=16, device=DEVICE,
        project=str(Path("models").resolve()), name=args.name, exist_ok=True,
        # The three channels are three moments in time, not real colors,
        # so don't let training shift hue or saturation.
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
        fliplr=0.5, flipud=0.5, mosaic=1.0, scale=0.3,
        # steadier training: a gentle learning rate that eases off smoothly
        optimizer="AdamW", lr0=0.0007, cos_lr=True, warmup_epochs=3, close_mosaic=10,
        save_period=5, patience=0, plots=True, verbose=False,
    )


if __name__ == "__main__":
    main()
