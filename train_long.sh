#!/bin/zsh
# Run detector training in the background so it keeps going after the terminal closes.
# The Mac is kept awake (caffeinate) until training finishes.
#
#   ./train_long.sh <name> <epochs> --train kick_01 ... synthetic --test kick_05
#   tail -f models/<name>.log            # watch progress
#
# If it stops early, resume with:
#   .venv/bin/python -c "from ultralytics import YOLO; YOLO('models/<name>/weights/last.pt').train(resume=True)"
cd "$(dirname "$0")"
name=$1; epochs=$2; shift 2
mkdir -p models
nohup caffeinate -i .venv/bin/python src/train_detector.py --name "$name" --epochs "$epochs" "$@" \
    > "models/$name.log" 2>&1 &
echo "training '$name' started (pid $!); log: models/$name.log"
