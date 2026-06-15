"""Разовый трекинг ствола по видео: считает кусты (уникальные track_id)."""
import sys
from pathlib import Path
from ultralytics import YOLO

weights = sys.argv[1]
video = sys.argv[2]
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
out_dir = sys.argv[4] if len(sys.argv) > 4 else "runs/vine_trunk/track"
tracker = str(Path("vinery/pipeline/botsort_vine.yaml").resolve())

model = YOLO(weights)
results = model.track(
    source=video, conf=conf, persist=True, tracker=tracker, save=True,
    project=str(Path(out_dir).parent), name=Path(out_dir).name,
    exist_ok=True, stream=True, verbose=False,
)

ids = set()
frames = with_det = 0
for r in results:
    frames += 1
    b = r.boxes
    if b is not None and b.id is not None and len(b):
        with_det += 1
        ids.update(b.id.int().tolist())

print(f"кадров: {frames}")
print(f"кадров со стволом: {with_det} ({100*with_det/max(frames,1):.0f}%)")
print(f"УНИКАЛЬНЫХ КУСТОВ (track_id): {len(ids)}")
print(f"видео с id -> {Path(out_dir).resolve()}")