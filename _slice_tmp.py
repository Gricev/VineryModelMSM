"""Разовая нарезка видео из папки 'Нарезка' в кадры (3 fps), без бирок рядов."""
import sys
from pathlib import Path
import cv2

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
TARGET_FPS = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0

videos = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"))
total = 0
for v in videos:
    cap = cv2.VideoCapture(str(v))
    if not cap.isOpened():
        print(f"!! не открыл {v.name}")
        continue
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(src_fps / TARGET_FPS))
    dst = OUT / v.stem
    dst.mkdir(parents=True, exist_ok=True)
    src_idx = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if src_idx % stride == 0:
            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok_enc:
                (dst / f"{v.stem}__f{saved:04d}.jpg").write_bytes(buf.tobytes())
                saved += 1
        src_idx += 1
    cap.release()
    total += saved
    print(f"{v.name}: {saved} кадров (src_fps {src_fps:.1f}, шаг {stride}) -> {dst}")
print(f"ИТОГО: {total} кадров")