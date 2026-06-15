"""SAM2-предразметка кадров ствола лозы (cam1) -> YOLO-боксы (vine_trunk).

SAM2 выдаёт МАСКИ, а детектору нужны БОКСЫ, поэтому здесь SAM2 работает как
авто-ПРЕДразметка: маска -> описанный прямоугольник -> строка YOLO. Результат —
ЧЕРНОВИК, его обязательно проверяют руками (в CVAT/Roboflow) перед обучением.

Когда что использовать (см. рекомендацию в обсуждении задачи):
  - ФОТО (отдельные кадры) -> этот модуль, режим `images`: промпт (бокс/точка)
    на каждый кадр -> маска -> бокс. Кадры независимы, протягивать нечего.
  - ВИДЕО проездов -> протяжка маски удобнее в CVAT/Roboflow со встроенным
    SAM2-трекером (готовый UI промпта + проверки). Тонкая экспериментальная
    функция `prelabel_video` оставлена для оффлайн-протяжки, но в проде по видео
    рекомендуется CVAT.

Выход кладётся в стейджинг `dataset/to_label/vine_trunk/` парами name.jpg+name.txt,
совместимо с `vinery.training.ingest import` (он перенесёт их в раскладку задачи
и допишет bush_index.csv). bush_code берётся из имени (ряд) — как и в ingest.

Тяжёлые зависимости (torch, sam2) импортируются ЛЕНИВО: конвертация маска->бокс
и её тест работают без них. SAM2 нужен только для самих предиктов — гоните на
машине с GPU (CPU-инференс SAM2, особенно видео, крайне медленный).

CLI:
    # фото из папки, ряд 3, интерактивно обводим ствол на каждом кадре
    python -m vinery.training.sam2_prelabel images photos/ --vineyard V1 --row 3 --interactive

    # фото с промптами из JSON (без GUI): {"IMG_001.jpg": {"box": [x0,y0,x1,y1]}}
    python -m vinery.training.sam2_prelabel images photos/ --vineyard V1 --row 3 --prompts p.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .ingest import DEFAULT_STAGING, IMAGE_EXTS, row_bush_code

# Чекпоинт по умолчанию: HF-идентификатор (from_pretrained сам скачает веса).
# large — точнее, base_plus/ tiny — быстрее. Под слабый GPU возьмите hiera-base-plus.
DEFAULT_MODEL = "facebook/sam2.1-hiera-large"
# Минимальная доля площади кадра, ниже которой маска считается мусором/бликом.
MIN_AREA_FRAC = 0.0005

try:
    import numpy as np
except ImportError:  # тест и конвертация работают и без numpy (медленный fallback)
    np = None


# ----------------------------------------------------------------- маска -> бокс
def mask_to_yolo_bbox(mask, img_w: int, img_h: int,
                      *, min_area_frac: float = MIN_AREA_FRAC):
    """2D-маску (truthy H x W) -> YOLO-бокс (cx, cy, w, h) в долях, или None.

    None, если маска пустая или меньше min_area_frac площади кадра. Работает и на
    numpy-массиве (быстрый путь), и на списке списков (медленный fallback для тестов).
    """
    if np is not None:
        m = np.asarray(mask, dtype=bool)
        ys, xs = np.where(m)
        if xs.size == 0 or xs.size < min_area_frac * img_w * img_h:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
    else:
        x0 = y0 = None
        x1 = y1 = 0
        area = 0
        for y, row in enumerate(mask):
            for x, v in enumerate(row):
                if v:
                    area += 1
                    x0 = x if x0 is None or x < x0 else x0
                    x1 = x if x > x1 else x1
                    y0 = y if y0 is None else y0
                    y1 = y
        if x0 is None or area < min_area_frac * img_w * img_h:
            return None

    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    cx = (x0 + bw / 2) / img_w
    cy = (y0 + bh / 2) / img_h
    return (cx, cy, bw / img_w, bh / img_h)


def yolo_line(bbox: tuple[float, float, float, float], cls: int = 0) -> str:
    return "%d %.6f %.6f %.6f %.6f" % (cls, *bbox)


# ----------------------------------------------------------------- загрузка SAM2
def _image_predictor(model: str, device: str | None):
    """Ленивая инициализация image-предиктора SAM2 (нужны torch+sam2)."""
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise RuntimeError(
            "Нужны torch и sam2. Установка: pip install torch && "
            "pip install 'git+https://github.com/facebookresearch/sam2.git'") from e
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        print("ВНИМАНИЕ: SAM2 на CPU очень медленный — используйте GPU.")
    return SAM2ImagePredictor.from_pretrained(model, device=dev)


def _read_rgb(path: Path):
    """Прочитать изображение как RGB-ndarray (через cv2, устойчиво к путям)."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("Нужен opencv-python для чтения кадров.") from e
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _pick_box(rgb):
    """Интерактивно обвести ствол мышью (cv2.selectROI). Возвращает [x0,y0,x1,y1] или None."""
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    r = cv2.selectROI("Обведи ствол (Enter=ок, c=пропустить)", bgr, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = r
    if w == 0 or h == 0:
        return None
    return [int(x), int(y), int(x + w), int(y + h)]


# ----------------------------------------------------------------- режим: фото
def _photo_name(vineyard: str, row: int, idx: int) -> str:
    """Имя кадра для отдельных фото: 'V1-R03__photos__f0007' (ряд как якорь)."""
    return f"{row_bush_code(vineyard, row)}__photos__f{idx:04d}"


def prelabel_images(
    images_dir: str | Path,
    *,
    vineyard: str,
    row: int,
    staging_root: str | Path = DEFAULT_STAGING,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    prompts: dict | None = None,
    interactive: bool = False,
    min_area_frac: float = MIN_AREA_FRAC,
    jpg_quality: int = 90,
) -> dict:
    """Предразметить папку фото стволов в стейджинг (name.jpg + name.txt).

    Промпт ствола на каждый кадр берётся (по приоритету): из `prompts`
    {filename: {"box":[x0,y0,x1,y1]}|{"point":[x,y]}}, иначе — интерактивно
    (`interactive`, окно cv2). Без промпта кадр пропускается (авто-детект тут
    ненадёжен). Возвращает сводку.
    """
    import cv2
    images_dir = Path(images_dir)
    files = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"Нет изображений в {images_dir}")

    out = Path(staging_root)
    out.mkdir(parents=True, exist_ok=True)
    predictor = _image_predictor(model, device)
    prompts = prompts or {}

    labeled = empty = skipped = 0
    for i, src in enumerate(files):
        rgb = _read_rgb(src)
        h, w = rgb.shape[:2]
        box = pts = None
        spec = prompts.get(src.name)
        if spec and "box" in spec:
            box = np.asarray(spec["box"], dtype=float)
        elif spec and "point" in spec:
            pts = np.asarray([spec["point"]], dtype=float)
        elif interactive:
            picked = _pick_box(rgb)
            if picked is None:
                skipped += 1
                continue
            box = np.asarray(picked, dtype=float)
        else:
            skipped += 1
            continue

        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=pts,
            point_labels=(np.ones(len(pts)) if pts is not None else None),
            box=box, multimask_output=False)
        bbox = mask_to_yolo_bbox(masks[0], w, h, min_area_frac=min_area_frac)

        name = _photo_name(vineyard, row, i)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        (out / f"{name}.jpg").write_bytes(buf.tobytes())
        label = yolo_line(bbox) + "\n" if bbox else ""   # пустой .txt = негатив
        (out / f"{name}.txt").write_text(label, encoding="utf-8")
        if bbox:
            labeled += 1
        else:
            empty += 1

    summary = {"labeled": labeled, "empty": empty, "skipped_no_prompt": skipped,
               "staging": str(out)}
    print("SAM2 предразметка фото:", summary)
    print("Дальше: проверьте боксы (CVAT/Roboflow), затем "
          "python -m vinery.training.ingest import --labels", out)
    return summary


# ----------------------------------------------------------------- режим: видео (экспериментально)
def prelabel_video(
    frames_dir: str | Path,
    prompts: list[dict],
    *,
    staging_root: str | Path = DEFAULT_STAGING,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    min_area_frac: float = MIN_AREA_FRAC,
) -> dict:
    """ЭКСПЕРИМЕНТАЛЬНО: оффлайн-протяжка SAM2 по кадрам прохода -> YOLO-боксы.

    frames_dir — кадры из `ingest extract` (стейджинг прохода);
    prompts — список объектов-стволов: [{"frame_idx":0,"obj_id":1,
              "box":[x0,y0,x1,y1]|"points":[[x,y]]}, ...].
    Каждый промпт = один ствол; SAM2 тянет его маску по видео, маски -> боксы.

    В проде по видео рекомендуется SAM2-трекер CVAT/Roboflow (готовый UI). Эта
    функция — для пакетной протяжки, когда промпты заранее известны.
    """
    try:
        import torch
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except ImportError as e:
        raise RuntimeError("Нужны torch и sam2 (см. prelabel_images).") from e

    frames_dir = Path(frames_dir)
    frame_files = sorted(p for p in frames_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTS)
    if not frame_files:
        raise FileNotFoundError(f"Нет кадров в {frames_dir}")

    # SAM2 init_state ждёт папку с кадрами, поименованными как индексы (0.jpg, ...).
    tmp = frames_dir / "_sam2_idx"
    tmp.mkdir(exist_ok=True)
    for i, f in enumerate(frame_files):
        (tmp / f"{i}.jpg").write_bytes(f.read_bytes())

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SAM2VideoPredictor.from_pretrained(model, device=dev)
    state = predictor.init_state(video_path=str(tmp))
    for pr in prompts:
        kw = {"box": np.asarray(pr["box"], dtype=float)} if "box" in pr else {
            "points": np.asarray(pr["points"], dtype=float),
            "labels": np.ones(len(pr["points"]))}
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=pr.get("frame_idx", 0),
            obj_id=pr["obj_id"], **kw)

    # frame_idx -> список боксов всех стволов на этом кадре
    boxes_per_frame: dict[int, list] = {}
    for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        import cv2  # noqa: F401  (читать размер кадра)
        h, w = _read_rgb(frame_files[fidx]).shape[:2]
        for k in range(len(obj_ids)):
            mask = (mask_logits[k] > 0.0).cpu().numpy().squeeze()
            bbox = mask_to_yolo_bbox(mask, w, h, min_area_frac=min_area_frac)
            if bbox:
                boxes_per_frame.setdefault(fidx, []).append(bbox)

    out = Path(staging_root)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, f in enumerate(frame_files):
        name = f.stem
        (out / f"{name}.jpg").write_bytes(f.read_bytes())
        lines = [yolo_line(b) for b in boxes_per_frame.get(i, [])]
        (out / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    summary = {"frames": written,
               "frames_with_boxes": len(boxes_per_frame), "staging": str(out)}
    print("SAM2 протяжка по видео:", summary)
    return summary


# ----------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(
        description="SAM2-предразметка стволов лозы (cam1) -> YOLO-боксы.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    im = sub.add_parser("images", help="предразметить папку отдельных фото")
    im.add_argument("images_dir")
    im.add_argument("--vineyard", default="V1")
    im.add_argument("--row", type=int, required=True)
    im.add_argument("--staging", default=DEFAULT_STAGING)
    im.add_argument("--model", default=DEFAULT_MODEL)
    im.add_argument("--device", default=None)
    im.add_argument("--prompts", default=None, help="JSON {file: {box|point}}")
    im.add_argument("--interactive", action="store_true", help="обводить ствол мышью")

    vi = sub.add_parser("video", help="ЭКСПЕРИМ.: протяжка по кадрам прохода")
    vi.add_argument("frames_dir")
    vi.add_argument("--prompts", required=True, help="JSON-список объектов-стволов")
    vi.add_argument("--staging", default=DEFAULT_STAGING)
    vi.add_argument("--model", default=DEFAULT_MODEL)
    vi.add_argument("--device", default=None)

    args = ap.parse_args()
    if args.cmd == "images":
        prompts = json.loads(Path(args.prompts).read_text("utf-8")) if args.prompts else None
        prelabel_images(args.images_dir, vineyard=args.vineyard, row=args.row,
                        staging_root=args.staging, model=args.model,
                        device=args.device, prompts=prompts, interactive=args.interactive)
    elif args.cmd == "video":
        prompts = json.loads(Path(args.prompts).read_text("utf-8"))
        prelabel_video(args.frames_dir, prompts, staging_root=args.staging,
                       model=args.model, device=args.device)


if __name__ == "__main__":
    main()
