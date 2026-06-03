"""Генератор СИНТЕТИЧЕСКИХ кадров ствола лозы + YOLO-разметка (камера 1).

Назначение: пока реальной разметки нет, дать «правдоподобные» данные, чтобы
прогнать весь конвейер обучения end-to-end (сборка датасета -> split-by-bush ->
yolo train) и убедиться, что трубопровод цел. Это НЕ замена реальным данным —
модель, обученная на этой синтетике, ничего полезного в поле не покажет.

Работает на чистом Python + Pillow (без numpy/torch), поэтому запускается в любом
окружении. Каждый «куст» получает свой стабильный вид ствола (положение/толщина/
цвет), а кадры одного куста лишь слегка дрожат — это имитирует то, что кадры
одного куста похожи, и проверяет, что split-by-bush реально защищает от утечки.

CLI:
    python -m vinery.training.synthetic --root dataset --rows 2 --bushes 8 --passes 2 --frames 6
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

from .yolo_dataset import INDEX_NAME, append_index, trunk_label_dir, trunk_src_dir

IMG_W, IMG_H = 640, 480


def _bush_code(vineyard: str, row: int, slot: int) -> str:
    return f"{vineyard}-R{row:02d}-B{slot:03d}"


def _draw_frame(rng: random.Random, trunk_cx: float, trunk_w: float,
                trunk_rgb: tuple[int, int, int]) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Нарисовать кадр со стволом и вернуть (изображение, YOLO-bbox в долях)."""
    # фон «листва»: зелёный с шумом
    img = Image.new("RGB", (IMG_W, IMG_H), (40 + rng.randint(0, 30), 90 + rng.randint(0, 40), 40))
    draw = ImageDraw.Draw(img)
    for _ in range(120):
        x, y = rng.randint(0, IMG_W), rng.randint(0, IMG_H)
        draw.point((x, y), fill=(rng.randint(20, 80), rng.randint(100, 160), rng.randint(20, 80)))

    # ствол: вертикальный прямоугольник во всю высоту, с лёгким дрожанием
    cx = min(max(trunk_cx + rng.uniform(-0.01, 0.01), 0.06), 0.94)
    w = max(trunk_w + rng.uniform(-0.005, 0.005), 0.02)
    x0 = int((cx - w / 2) * IMG_W)
    x1 = int((cx + w / 2) * IMG_W)
    y0 = int(0.10 * IMG_H + rng.uniform(-5, 5))
    y1 = int(0.98 * IMG_H + rng.uniform(-5, 5))
    draw.rectangle([x0, y0, x1, y1], fill=trunk_rgb)

    bbox = (cx, (y0 + y1) / 2 / IMG_H, w, (y1 - y0) / IMG_H)
    return img, bbox


def generate(root: str | Path = "dataset", *, vineyard: str = "V1",
             rows: int = 2, bushes_per_row: int = 8, passes: int = 2,
             frames_per_pass: int = 6, seed: int = 0) -> int:
    """Сгенерировать синтетику в dataset/annotations/vine_trunk_detection/.

    Возвращает число записанных кадров. Раскладка:
        annotations/vine_trunk_detection/
            images/<name>.jpg
            labels/<name>.txt        # YOLO: "0 cx cy w h"
            bush_index.csv           # name,bush_code  (для split-by-bush)
    name = "<bush_code>__pass_<NN>__f<NNNN>".
    """
    rng = random.Random(seed)
    img_dir = trunk_src_dir(root) / "images"
    lbl_dir = trunk_label_dir(root)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    index: list[tuple[str, str]] = []
    n = 0
    for row in range(1, rows + 1):
        for slot in range(1, bushes_per_row + 1):
            code = _bush_code(vineyard, row, slot)
            # стабильный «вид» куста: фиксируем по его коду
            brng = random.Random(f"{seed}:{code}")
            base_cx = brng.uniform(0.30, 0.70)
            base_w = brng.uniform(0.05, 0.12)
            rgb = (brng.randint(70, 110), brng.randint(45, 75), brng.randint(25, 45))
            for p in range(1, passes + 1):
                for f in range(frames_per_pass):
                    img, bbox = _draw_frame(rng, base_cx, base_w, rgb)
                    name = f"{code}__pass_{p:02d}__f{f:04d}"
                    img.save(img_dir / f"{name}.jpg", quality=85)
                    (lbl_dir / f"{name}.txt").write_text(
                        "0 %.6f %.6f %.6f %.6f\n" % bbox, encoding="utf-8")
                    index.append((name, code))
                    n += 1

    append_index(root, index)
    print(f"Сгенерировано {n} кадров, {rows * bushes_per_row} кустов -> {trunk_src_dir(root)}")
    print(f"Индекс куст<->кадр: {trunk_src_dir(root) / INDEX_NAME}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Генератор синтетики стволов лозы (cam1).")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--vineyard", default="V1")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--bushes", type=int, default=8, dest="bushes_per_row")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--frames", type=int, default=6, dest="frames_per_pass")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.root, vineyard=args.vineyard, rows=args.rows,
             bushes_per_row=args.bushes_per_row, passes=args.passes,
             frames_per_pass=args.frames_per_pass, seed=args.seed)


if __name__ == "__main__":
    main()