"""SAM2-предразметка кроны (cam2) -> YOLO-seg полигоны (слой ОРГАНА).

В отличие от cam1 (`sam2_prelabel`, маска -> БОКС), cam2 — СЕГМЕНТАЦИЯ: SAM2 даёт
маску органа -> ПОЛИГОН -> строка YOLO-seg. Результат — ЧЕРНОВИК слоя ОРГАНА
(`leaf`/`inflorescence`/`cluster`) под ОБЯЗАТЕЛЬНУЮ ручную правку: разметчик
проверяет контур и доразмечает слой ПОРАЖЕНИЯ + коды болезней (CANOPY_LABELS.md §2).
severity тут не считается — это рантайм (CanopyAnalyzer).

Класс органа выводится из ФЕНОФАЗЫ партии (партия = одна фенофаза, см. CANOPY_LABELS.md):
    leaf -> leaf · flowering -> inflorescence · cluster -> cluster.

Листья — МНОГО инстансов в кадре, поэтому кроме промптовых режимов (как в cam1) есть
автоматический (`auto`, SAM2AutomaticMaskGenerator): он предлагает все маски-кандидаты,
из них лепятся полигоны органа (шумный черновик — чистится руками, мусор/фон удаляется).

Выход -> стейджинг фенофазы `dataset/to_label/canopy/<phase>/` парами name.jpg+name.txt
(YOLO-seg), совместимо с `ingest canopy-import` (он перенесёт в annotations/canopy_seg
и допишет canopy_index.csv). bush_code берётся из имени (ряд) — split по ряду без правок.

Тяжёлые зависимости (torch, sam2) импортируются ЛЕНИВО: конвертация маска->полигон
и её тест работают без них (нужен только cv2+numpy). SAM2 нужен лишь для самих
предиктов — гоните на GPU (CPU-инференс, особенно auto, крайне медленный).

CLI:
    # авто: предложить полигоны всех листьев на каждом фото (черновик), ряд 3
    python -m vinery.training.sam2_canopy auto photos/ --vineyard V1 --row 3 --phase leaf

    # промпты из JSON (без GUI): {"IMG_001.jpg": {"boxes": [[x0,y0,x1,y1], ...]}}
    python -m vinery.training.sam2_canopy images photos/ --row 3 --phase leaf --prompts p.json

    # интерактивно: обводить по листу за раз, пустой ROI = следующий кадр
    python -m vinery.training.sam2_canopy images photos/ --row 3 --phase leaf --interactive
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingest import (
    DEFAULT_CANOPY_STAGING, IMAGE_EXTS, canopy_staging_dir, row_bush_code,
)
from .sam2_prelabel import (
    DEFAULT_MODEL, MIN_AREA_FRAC, _image_predictor, _pick_box, _read_rgb,
)
from .yolo_dataset import CANOPY_SEG_CLASSES

try:
    import numpy as np
except ImportError:  # конвертация требует numpy+cv2; тест гоняем с ними
    np = None

# Фенофаза -> класс ОРГАНА (слой 1). Партия = одна фенофаза, поэтому орган однозначен.
PHASE_ORGAN = {"leaf": "leaf", "flowering": "inflorescence", "cluster": "cluster"}
# Доля периметра контура для упрощения полигона (approxPolyDP). Больше -> грубее полигон.
DEFAULT_EPSILON_FRAC = 0.004


def organ_class_id(phase: str) -> int:
    """id класса органа для фенофазы в CANOPY_SEG_CLASSES (leaf->0, infl->1, cluster->2)."""
    try:
        organ = PHASE_ORGAN[phase]
    except KeyError:
        raise ValueError(
            f"Неизвестная фенофаза {phase!r}; допустимо: {', '.join(PHASE_ORGAN)}")
    return CANOPY_SEG_CLASSES.index(organ)


# ----------------------------------------------------------------- маска -> полигон
def mask_to_yolo_polygon(mask, img_w: int, img_h: int, *,
                         min_area_frac: float = MIN_AREA_FRAC,
                         epsilon_frac: float = DEFAULT_EPSILON_FRAC):
    """2D-маску (truthy H x W) -> нормированный полигон [x1,y1,x2,y2,...] или None.

    Берётся ВНЕШНИЙ контур наибольшей связной области, упрощается approxPolyDP и
    нормируется в доли [0..1]. None, если маска пустая, мельче min_area_frac площади
    кадра или вырождена (<3 вершин). Требует cv2+numpy (полигонизация без них не делается).
    """
    if np is None:
        raise RuntimeError("Нужен numpy для полигонизации маски.")
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("Нужен opencv-python для полигонизации маски.") from e

    m = np.asarray(mask, dtype=np.uint8)
    if m.ndim != 2 or not m.any():
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area_frac * img_w * img_h:
        return None
    eps = epsilon_frac * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    poly: list[float] = []
    for x, y in approx:
        poly.append(min(max(float(x) / img_w, 0.0), 1.0))
        poly.append(min(max(float(y) / img_h, 0.0), 1.0))
    return poly


def yolo_seg_line(poly_norm: list[float], cls: int) -> str:
    """Строка YOLO-seg: '<cls> x1 y1 x2 y2 ...' (нормированные координаты)."""
    return f"{cls} " + " ".join("%.6f" % v for v in poly_norm)


# ----------------------------------------------------------------- имена / запись
def _canopy_photo_name(vineyard: str, row: int, phase: str, idx: int) -> str:
    """Имя отдельного фото cam2: 'V1-R03__leaf__f0007' (как в ingest.stage_canopy_images)."""
    return f"{row_bush_code(vineyard, row)}__{phase}__f{idx:04d}"


def _write_pair(out_dir: Path, name: str, rgb, lines: list[str], jpg_quality: int) -> None:
    """Записать кадр (jpg) и его YOLO-seg разметку (txt; пустой = негатив/healthy)."""
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    if ok:
        (out_dir / f"{name}.jpg").write_bytes(buf.tobytes())
    (out_dir / f"{name}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ----------------------------------------------------------------- промпты -> маски
def _masks_from_prompts(predictor, rgb, spec: dict) -> list:
    """Получить маски органа по промптам кадра: spec={"boxes":[...], "points":[[x,y],...]}.

    Каждый бокс/точка = один инстанс органа (multimask_output=False). Возвращает
    список 2D-масок (bool). Пустой spec -> [].
    """
    predictor.set_image(rgb)
    masks: list = []
    for box in spec.get("boxes", []):
        m, _, _ = predictor.predict(box=np.asarray(box, dtype=float),
                                    multimask_output=False)
        masks.append(m[0])
    for pt in spec.get("points", []):
        m, _, _ = predictor.predict(point_coords=np.asarray([pt], dtype=float),
                                    point_labels=np.ones(1), multimask_output=False)
        masks.append(m[0])
    return masks


def _pick_boxes(rgb) -> list:
    """Интерактивно обвести НЕСКОЛЬКО органов (по одному за раз). Пустой ROI завершает."""
    boxes = []
    while True:
        box = _pick_box(rgb)   # окно cv2.selectROI (Enter=ок, c/пустой=стоп)
        if box is None:
            break
        boxes.append(box)
    return boxes


def _auto_mask_generator(model: str, device: str | None):
    """Ленивая инициализация SAM2AutomaticMaskGenerator (нужны torch+sam2)."""
    try:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as e:
        raise RuntimeError(
            "Нужны torch и sam2 для auto-режима (см. sam2_prelabel).") from e
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        print("ВНИМАНИЕ: SAM2 auto на CPU крайне медленный — используйте GPU.")
    return SAM2AutomaticMaskGenerator.from_pretrained(model, device=dev)


# ----------------------------------------------------------------- предразметка фото
def prelabel_canopy_images(
    images_dir: str | Path,
    *,
    vineyard: str,
    row: int,
    phase: str = "leaf",
    staging_root: str | Path = DEFAULT_CANOPY_STAGING,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    prompts: dict | None = None,
    interactive: bool = False,
    auto: bool = False,
    min_area_frac: float = MIN_AREA_FRAC,
    max_instances: int = 60,
    epsilon_frac: float = DEFAULT_EPSILON_FRAC,
    jpg_quality: int = 90,
) -> dict:
    """Предразметить папку фото кроны в стейджинг фенофазы (name.jpg + name.txt, YOLO-seg).

    Источник масок органа на кадр (по приоритету):
      auto=True       — SAM2AutomaticMaskGenerator предлагает все маски (черновик);
      prompts[file]   — {"boxes":[...], "points":[...]}: по инстансу на промпт;
      interactive=True— обводить органы мышью (по одному за раз).
    Все маски кадра -> полигоны класса фенофазы (organ_class_id). severity не считается.
    Пустой кадр пишется негативом (.txt пуст) — healthy-органы тоже нужны (CANOPY_LABELS.md §2).
    """
    organ_cls = organ_class_id(phase)
    images_dir = Path(images_dir)
    files = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"Нет изображений в {images_dir}")

    out = canopy_staging_dir(staging_root, phase)
    out.mkdir(parents=True, exist_ok=True)
    prompts = prompts or {}

    predictor = None if auto else _image_predictor(model, device)
    amg = _auto_mask_generator(model, device) if auto else None

    frames = instances = empty = 0
    for i, src in enumerate(files):
        rgb = _read_rgb(src)
        h, w = rgb.shape[:2]

        if auto:
            raw = amg.generate(rgb)
            raw.sort(key=lambda d: d.get("area", 0), reverse=True)
            masks = [d["segmentation"] for d in raw[:max_instances]]
        elif src.name in prompts:
            masks = _masks_from_prompts(predictor, rgb, prompts[src.name])
        elif interactive:
            boxes = _pick_boxes(rgb)
            masks = _masks_from_prompts(predictor, rgb, {"boxes": boxes})
        else:
            masks = []   # кадр без промпта -> негатив

        lines = []
        for m in masks:
            poly = mask_to_yolo_polygon(m, w, h, min_area_frac=min_area_frac,
                                        epsilon_frac=epsilon_frac)
            if poly:
                lines.append(yolo_seg_line(poly, organ_cls))

        name = _canopy_photo_name(vineyard, row, phase, i)
        _write_pair(out, name, rgb, lines, jpg_quality)
        frames += 1
        instances += len(lines)
        if not lines:
            empty += 1

    summary = {"frames": frames, "instances": instances, "empty_frames": empty,
               "phase": phase, "organ_class": CANOPY_SEG_CLASSES[organ_cls],
               "staging": str(out)}
    print("SAM2 предразметка кроны (черновик слоя органа):", summary)
    print("Дальше: проверьте/поправьте полигоны и доразметьте слой ПОРАЖЕНИЯ "
          "(CANOPY_LABELS.md), затем:")
    print(f"  python -m vinery.training.ingest canopy-import --labels {out}")
    return summary


# ----------------------------------------------------------------- CLI
def _add_common(p) -> None:
    p.add_argument("images_dir")
    p.add_argument("--vineyard", default="V1")
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--phase", default="leaf", choices=list(PHASE_ORGAN),
                   help="фенофаза партии -> класс органа (по умолч. leaf)")
    p.add_argument("--staging", default=DEFAULT_CANOPY_STAGING)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None)
    p.add_argument("--min-area-frac", type=float, default=MIN_AREA_FRAC, dest="min_area_frac")
    p.add_argument("--epsilon-frac", type=float, default=DEFAULT_EPSILON_FRAC, dest="epsilon_frac")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SAM2-предразметка кроны (cam2) -> YOLO-seg полигоны органа.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    au = sub.add_parser("auto", help="авто: предложить полигоны всех органов (черновик)")
    _add_common(au)
    au.add_argument("--max-instances", type=int, default=60, dest="max_instances")

    im = sub.add_parser("images", help="промптовая предразметка (boxes/points или мышью)")
    _add_common(im)
    im.add_argument("--prompts", default=None,
                    help='JSON {file: {"boxes":[[x0,y0,x1,y1],...], "points":[[x,y],...]}}')
    im.add_argument("--interactive", action="store_true", help="обводить органы мышью")

    args = ap.parse_args()
    if args.cmd == "auto":
        prelabel_canopy_images(
            args.images_dir, vineyard=args.vineyard, row=args.row, phase=args.phase,
            staging_root=args.staging, model=args.model, device=args.device,
            auto=True, min_area_frac=args.min_area_frac, max_instances=args.max_instances,
            epsilon_frac=args.epsilon_frac)
    elif args.cmd == "images":
        prompts = json.loads(Path(args.prompts).read_text("utf-8")) if args.prompts else None
        prelabel_canopy_images(
            args.images_dir, vineyard=args.vineyard, row=args.row, phase=args.phase,
            staging_root=args.staging, model=args.model, device=args.device,
            prompts=prompts, interactive=args.interactive,
            min_area_frac=args.min_area_frac, epsilon_frac=args.epsilon_frac)


if __name__ == "__main__":
    main()