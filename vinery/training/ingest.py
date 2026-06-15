"""Ингест РЕАЛЬНЫХ видео камеры 1 в датасет детекции ствола лозы (vine_trunk).

В отличие от `synthetic.py` (рисует кадры сам), этот модуль берёт настоящие mp4
проездов тележки и готовит их к разметке во внешнем инструменте (CVAT/Roboflow),
а затем принимает YOLO-экспорт обратно в раскладку задачи.

Двухэтапный поток:

    1) extract  — видео прохода -> кадры в стейджинг `dataset/to_label/vine_trunk/`
                  (+ ingest_manifest.csv с провенансом). Эти кадры грузятся в CVAT.
    2) import   — YOLO-экспорт из CVAT (кадры + .txt боксы) -> раскладка задачи
                  `dataset/annotations/vine_trunk_detection/{images,labels}` и
                  дописывает bush_index.csv. Дальше — yolo_dataset.py / train_trunk.py.

Привязка к кусту на сыром видео ещё неизвестна (это работа пайплайна-локализатора),
поэтому кадры привязываются к РЯДУ: bush_code = "V1-R03". Этого достаточно для
рекомендованного split-by-row (group_by='row') — кадры одного ряда не растекутся
между train/val (см. splits.py, DATASET.md §«split по кусту»).

cv2 импортируется ЛЕНИВО (как в pipeline/stream.py): `import` работает без opencv,
он только копирует файлы; cv2 нужен лишь для `extract` (декодирование видео).

Камера 2 (лист/соцветие/гроздь) — отдельная дорожка: те же стейджинг-команды с
префиксом `canopy-`, но раскладка по подпапке ФЕНОФАЗЫ (`canopy/<phase>`) и фенофаза
в имени кадра. Размечается МАСКАМИ (YOLO-seg) по таксономии CANOPY_LABELS.md, а не
классом `vine_trunk`. Анти-утечку (split по ряду) и ядро декодирования делит с cam1.
Приём cam2-экспорта — `canopy-import`: YOLO-seg экспорт (кадры + .txt полигоны) ->
раскладка задачи `annotations/canopy_seg/{images,labels}` + `canopy_index.csv`
(name,bush_code,phase). Дальше — yolo_dataset.py --task canopy. severity из масок тут
не считается (это рантайм CanopyAnalyzer); приём — чистый роутинг файлов, как у cam1.

CLI:
    # cam1: нарезать кадры из видео прохода (row 3, проход 01) на разметку
    python -m vinery.training.ingest extract video.mp4 --vineyard V1 --row 3 --pass 1 --fps 3

    # cam1: забрать YOLO-экспорт из CVAT обратно в датасет
    python -m vinery.training.ingest import --labels cvat_export/obj_train_data

    # cam2: нарезать кадры цветения (партия = одна фенофаза) в стейджинг canopy/flowering
    python -m vinery.training.ingest canopy-extract video.mp4 --row 3 --phase flowering --pass 1

    # cam2: забрать YOLO-seg экспорт (полигоны кроны) обратно в датасет
    python -m vinery.training.ingest canopy-import --labels seg_export/labels
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from .yolo_dataset import (
    CANOPY_INDEX_NAME, INDEX_NAME, canopy_label_dir, canopy_src_dir,
    trunk_label_dir, trunk_src_dir,
)

try:
    import cv2
except ImportError:  # позволяет делать import-этап без установленного opencv
    cv2 = None

DEFAULT_STAGING = "dataset/to_label/vine_trunk"
MANIFEST_NAME = "ingest_manifest.csv"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# cam2 (лист/соцветие/гроздь): отдельная дорожка, стейджинг по фенофазе И ряду.
# Таксономию и геометрию (маски YOLO-seg) см. CANOPY_LABELS.md; раскладку/анти-утечку
# (split по ряду) переиспользуем из cam1. Партия = одна фенофаза (диктуется датой съёмки).
DEFAULT_CANOPY_STAGING = "dataset/to_label/canopy"
CANOPY_PHASES = ("flowering", "cluster", "leaf")


# ----------------------------------------------------------------- имена / коды
def row_bush_code(vineyard: str, row: int) -> str:
    """Код РЯДА в роли bush_code на сыром видео: ('V1', 3) -> 'V1-R03'.

    Точный куст из сырого видео неизвестен; ряд — самый надёжный якорь для
    split-by-row. row_of('V1-R03') == 'V1-R03', так что группировка не ломается.
    """
    return f"{vineyard}-R{row:02d}"


def frame_name(vineyard: str, row: int, pass_no: int, idx: int) -> str:
    """Имя кадра по соглашению датасета: 'V1-R03__pass_01__f0007'."""
    return f"{row_bush_code(vineyard, row)}__pass_{pass_no:02d}__f{idx:04d}"


def bush_code_from_name(name: str) -> str:
    """Достать bush_code из имени кадра: всё до '__'.

    'V1-R03__pass_01__f0007' -> 'V1-R03'  (ряд, реальные кадры);
    'V1-R01-B001__pass_01__f0000' -> 'V1-R01-B001'  (синтетика, полный куст).
    """
    return name.split("__", 1)[0]


# ------------------------------------------------------- общее ядро декодирования
def _open_video(video: str | Path, fps: float | None, every: int | None) -> tuple:
    """Открыть видео и посчитать шаг подвыборки. Возвращает (cap, src_fps, stride).

    Подвыборка: `every` (каждый N-й кадр) приоритетнее; иначе `fps` задаёт целевую
    частоту (шаг = round(fps_видео / fps)). cv2 проверяется лениво — общий вход
    для extract обеих камер.
    """
    if cv2 is None:
        raise RuntimeError(
            "Для extract нужен opencv-python: "
            ".venv\\Scripts\\python.exe -m pip install opencv-python numpy")
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(f"Нет видео: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = every if every else max(1, round(src_fps / fps)) if fps else 1
    return cap, src_fps, stride


def _read_subsampled(cap, stride: int, max_frames: int | None):
    """Итерировать кадры видео с шагом stride, отдавая (saved_idx, src_idx, image)."""
    src_idx = saved = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if src_idx % stride == 0:
            yield saved, src_idx, image
            saved += 1
            if max_frames and saved >= max_frames:
                break
        src_idx += 1


def _write_jpg(path: Path, image, quality: int) -> bool:
    """Записать кадр в JPG через imencode+write_bytes (устойчиво к не-ASCII путям)."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        path.write_bytes(buf.tobytes())
    return ok


# ----------------------------------------------------------------- этап 1: extract
def extract_frames(
    video: str | Path,
    *,
    vineyard: str,
    row: int,
    pass_no: int,
    staging_root: str | Path = DEFAULT_STAGING,
    fps: float | None = 3.0,
    every: int | None = None,
    max_frames: int | None = None,
    jpg_quality: int = 90,
) -> int:
    """Нарезать кадры из видео прохода cam1 в стейджинг-папку для разметки.

    Подвыборка см. `_open_video`. Возвращает число кадров. Провенанс (откуда кадр)
    дописывается в ingest_manifest.csv.
    """
    cap, src_fps, stride = _open_video(video, fps, every)
    video = Path(video)
    out_dir = Path(staging_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[tuple] = []
    for saved, src_idx, image in _read_subsampled(cap, stride, max_frames):
        name = frame_name(vineyard, row, pass_no, saved)
        if _write_jpg(out_dir / f"{name}.jpg", image, jpg_quality):
            manifest_rows.append(
                (name, video.name, vineyard, row, pass_no,
                 src_idx, round(src_idx / src_fps, 3)))
    cap.release()

    _append_manifest(out_dir, manifest_rows)
    print(f"{video.name}: извлечено {len(manifest_rows)} кадров "
          f"(шаг {stride}, src_fps {src_fps:.1f}) -> {out_dir}")
    print(f"Дальше: загрузите кадры в CVAT, разметьте класс 'vine_trunk', "
          f"экспортируйте в YOLO и запустите 'import'.")
    return len(manifest_rows)


def _append_manifest(out_dir: Path, rows: list[tuple]) -> None:
    path = out_dir / MANIFEST_NAME
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "video", "vineyard", "row", "pass",
                        "src_frame_index", "t_sec"])
        w.writerows(rows)


# ----------------------------------------------------------------- этап 1b: отдельные фото
def stage_images(
    images_dir: str | Path,
    *,
    vineyard: str,
    row: int,
    staging_root: str | Path = DEFAULT_STAGING,
    tag: str = "photos",
) -> int:
    """Скопировать отдельные ФОТО в стейджинг с именами по конвенции (бирка ряда).

    Без cv2/SAM2 — простое копирование с переименованием. Нужно, чтобы у фото
    появилась бирка ряда в имени (V1-R03__photos__f0007), иначе `import` присвоит
    каждому фото свою группу по исходному имени (IMG_...). Возвращает число фото.
    """
    images_dir = Path(images_dir)
    files = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"Нет изображений в {images_dir}")
    out = Path(staging_root)
    out.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        name = f"{row_bush_code(vineyard, row)}__{tag}__f{i:04d}"
        shutil.copy2(src, out / f"{name}.jpg")
    print(f"{images_dir}: уложено {len(files)} фото -> {out} (бирка {row_bush_code(vineyard, row)})")
    return len(files)


# ----------------------------------------------------- стейджинг cam2 (canopy)
def _check_phase(phase: str) -> str:
    if phase not in CANOPY_PHASES:
        raise ValueError(
            f"Неизвестная фенофаза {phase!r}; допустимо: {', '.join(CANOPY_PHASES)}")
    return phase


def canopy_staging_dir(staging_root: str | Path, phase: str) -> Path:
    """Подпапка стейджинга cam2 по фенофазе: dataset/to_label/canopy/flowering."""
    return Path(staging_root) / _check_phase(phase)


def canopy_frame_name(vineyard: str, row: int, phase: str,
                      pass_no: int, idx: int) -> str:
    """Имя кадра cam2: 'V1-R03__flowering__pass_01__f0007'.

    Бирка РЯДА — первый сегмент (как в cam1), поэтому `bush_code_from_name` и
    split-by-row работают без изменений. Фенофаза — второй сегмент (восстановима
    через `canopy_phase_from_name`): маршрутизирует кадр в нужную задачу разметки.
    """
    return (f"{row_bush_code(vineyard, row)}__{_check_phase(phase)}"
            f"__pass_{pass_no:02d}__f{idx:04d}")


def canopy_phase_from_name(name: str) -> str | None:
    """Достать фенофазу из имени кадра cam2: второй сегмент ('V1-R03__flowering__...')."""
    parts = name.split("__")
    return parts[1] if len(parts) >= 3 and parts[1] in CANOPY_PHASES else None


def extract_canopy_frames(
    video: str | Path,
    *,
    vineyard: str,
    row: int,
    phase: str,
    pass_no: int,
    staging_root: str | Path = DEFAULT_CANOPY_STAGING,
    fps: float | None = 2.0,
    every: int | None = None,
    max_frames: int | None = None,
    jpg_quality: int = 90,
) -> int:
    """Нарезать кадры видео cam2 в стейджинг по фенофазе (canopy/<phase>).

    То же ядро, что у cam1 `extract_frames`, но кадры раскладываются в подпапку
    фенофазы и имя несёт <phase> вторым сегментом. Размечаются МАСКАМИ (YOLO-seg)
    по таксономии CANOPY_LABELS.md — не `vine_trunk`. Провенанс (+столбец phase)
    дописывается в ingest_manifest.csv внутри подпапки фенофазы.
    """
    _check_phase(phase)
    cap, src_fps, stride = _open_video(video, fps, every)
    video = Path(video)
    out_dir = canopy_staging_dir(staging_root, phase)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[tuple] = []
    for saved, src_idx, image in _read_subsampled(cap, stride, max_frames):
        name = canopy_frame_name(vineyard, row, phase, pass_no, saved)
        if _write_jpg(out_dir / f"{name}.jpg", image, jpg_quality):
            manifest_rows.append(
                (name, video.name, vineyard, row, phase, pass_no,
                 src_idx, round(src_idx / src_fps, 3)))
    cap.release()

    _append_canopy_manifest(out_dir, manifest_rows)
    print(f"{video.name}: извлечено {len(manifest_rows)} кадров cam2 [{phase}] "
          f"(шаг {stride}, src_fps {src_fps:.1f}) -> {out_dir}")
    print("Дальше: загрузите кадры в CVAT/Roboflow, разметьте МАСКАМИ по "
          "CANOPY_LABELS.md, экспортируйте в YOLO-seg.")
    return len(manifest_rows)


def stage_canopy_images(
    images_dir: str | Path,
    *,
    vineyard: str,
    row: int,
    phase: str,
    staging_root: str | Path = DEFAULT_CANOPY_STAGING,
) -> int:
    """Скопировать отдельные ФОТО cam2 в стейджинг фенофазы с биркой ряда+фенофазы.

    Аналог cam1 `stage_images`, но раскладка по подпапке фенофазы и фенофаза в
    имени (второй сегмент). Без cv2 — простое копирование с переименованием.
    Имя 'V1-R03__flowering__f0007' (без pass — у отдельных фото прохода нет).
    """
    _check_phase(phase)
    images_dir = Path(images_dir)
    files = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"Нет изображений в {images_dir}")
    out = canopy_staging_dir(staging_root, phase)
    out.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        name = f"{row_bush_code(vineyard, row)}__{phase}__f{i:04d}"
        shutil.copy2(src, out / f"{name}.jpg")
    print(f"{images_dir}: уложено {len(files)} фото cam2 [{phase}] -> {out} "
          f"(бирка {row_bush_code(vineyard, row)})")
    return len(files)


def _append_canopy_manifest(out_dir: Path, rows: list[tuple]) -> None:
    path = out_dir / MANIFEST_NAME
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "video", "vineyard", "row", "phase", "pass",
                        "src_frame_index", "t_sec"])
        w.writerows(rows)


# ----------------------------------------------------------------- этап 2: import
def _index_names(root: str | Path) -> set[str]:
    """Имена, уже записанные в bush_index.csv (для идемпотентного дозаписывания)."""
    path = trunk_src_dir(root) / INDEX_NAME
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row["name"] for row in csv.DictReader(f)}


def _find_image(images_dir: Path, name: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / f"{name}{ext}"
        if p.exists():
            return p
    return None


def import_labels(
    labels_dir: str | Path,
    *,
    images_dir: str | Path | None = None,
    dataset_root: str | Path = "dataset",
    keep_empty: bool = True,
) -> dict:
    """Принять YOLO-экспорт из CVAT в раскладку задачи vine_trunk_detection.

    labels_dir — папка с YOLO .txt (напр. cvat_export/obj_train_data);
    images_dir — папка с кадрами (по умолчанию стейджинг DEFAULT_STAGING);
    keep_empty — оставлять кадры без боксов как негативные сэмплы (полезно детектору).

    Для каждого .txt с парной картинкой: копирует кадр в images/, разметку в
    labels/, дописывает (name, bush_code) в bush_index.csv (без дублей).
    bush_code берётся из имени кадра (`bush_code_from_name`) — так же режется split.
    """
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir) if images_dir else Path(DEFAULT_STAGING)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Нет папки разметки: {labels_dir}")

    dst_img = trunk_src_dir(dataset_root) / "images"
    dst_lbl = trunk_label_dir(dataset_root)
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    already = _index_names(dataset_root)
    new_rows: list[tuple[str, str]] = []
    imported = skipped_no_image = skipped_empty = 0

    for lbl in sorted(labels_dir.glob("*.txt")):
        name = lbl.stem
        if name in ("classes", "obj"):     # служебные файлы YOLO-экспорта
            continue
        img = _find_image(images_dir, name)
        if img is None:
            skipped_no_image += 1
            continue
        if not keep_empty and lbl.stat().st_size == 0:
            skipped_empty += 1
            continue

        shutil.copy2(img, dst_img / f"{name}.jpg")
        shutil.copy2(lbl, dst_lbl / f"{name}.txt")
        if name not in already:
            new_rows.append((name, bush_code_from_name(name)))
            already.add(name)
        imported += 1

    _append_index_dedup(dataset_root, new_rows)
    summary = {
        "imported": imported,
        "new_index_rows": len(new_rows),
        "skipped_no_image": skipped_no_image,
        "skipped_empty": skipped_empty,
        "images_dir": str(trunk_src_dir(dataset_root) / "images"),
    }
    print("Импорт завершён:", summary)
    if skipped_no_image:
        print(f"  ВНИМАНИЕ: {skipped_no_image} .txt без парной картинки в {images_dir} "
              f"(укажите --images, если кадры лежат в другом месте).")
    print("Дальше: python -m vinery.training.yolo_dataset --root dataset --group-by row")
    return summary


def _append_index_dedup(root: str | Path, rows: list[tuple[str, str]]) -> None:
    """Дописать (name, bush_code) в bush_index.csv, создав заголовок при нужде."""
    if not rows:
        return
    path = trunk_src_dir(root) / INDEX_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "bush_code"])
        w.writerows(rows)


# -------------------------------------------------- этап 2 (cam2): приём масок
def _canopy_index_names(root: str | Path) -> set[str]:
    """Имена, уже записанные в canopy_index.csv (для идемпотентного дозаписывания)."""
    path = canopy_src_dir(root) / CANOPY_INDEX_NAME
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row["name"] for row in csv.DictReader(f)}


def _find_image_recursive(images_dir: Path, name: str) -> Path | None:
    """Найти кадр по имени где угодно под images_dir (стейджинг cam2 вложен по фенофазе)."""
    for ext in IMAGE_EXTS:
        hit = next(images_dir.rglob(f"{name}{ext}"), None)
        if hit is not None:
            return hit
    return None


def import_canopy_labels(
    labels_dir: str | Path,
    *,
    images_dir: str | Path | None = None,
    dataset_root: str | Path = "dataset",
    keep_empty: bool = True,
) -> dict:
    """Принять YOLO-seg экспорт кроны в раскладку задачи canopy_seg.

    Зеркало import_labels (cam1), но для полигонов: .txt с разметкой YOLO-seg
    ('<cls> x1 y1 x2 y2 ...') копируются как есть, картинки ищутся РЕКУРСИВНО
    (стейджинг cam2 вложен по фенофазе). Дописывает (name, bush_code, phase) в
    canopy_index.csv: bush_code из имени (split по ряду), phase вторым сегментом
    имени (`canopy_phase_from_name`). severity тут не считается — это рантайм.
    keep_empty — оставлять кадры без масок негативами (нужны, чтобы модель учила
    'орган = болезнь' через healthy-органы, см. CANOPY_LABELS.md §2).
    """
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir) if images_dir else Path(DEFAULT_CANOPY_STAGING)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Нет папки разметки: {labels_dir}")

    dst_img = canopy_src_dir(dataset_root) / "images"
    dst_lbl = canopy_label_dir(dataset_root)
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    already = _canopy_index_names(dataset_root)
    new_rows: list[tuple[str, str, str]] = []
    imported = skipped_no_image = skipped_empty = 0

    for lbl in sorted(labels_dir.rglob("*.txt")):
        name = lbl.stem
        if name in ("classes", "obj"):     # служебные файлы YOLO-экспорта
            continue
        img = _find_image_recursive(images_dir, name)
        if img is None:
            skipped_no_image += 1
            continue
        if not keep_empty and lbl.stat().st_size == 0:
            skipped_empty += 1
            continue

        shutil.copy2(img, dst_img / f"{name}.jpg")
        shutil.copy2(lbl, dst_lbl / f"{name}.txt")
        if name not in already:
            new_rows.append((name, bush_code_from_name(name),
                             canopy_phase_from_name(name) or ""))
            already.add(name)
        imported += 1

    _append_canopy_index_dedup(dataset_root, new_rows)
    summary = {
        "imported": imported,
        "new_index_rows": len(new_rows),
        "skipped_no_image": skipped_no_image,
        "skipped_empty": skipped_empty,
        "images_dir": str(canopy_src_dir(dataset_root) / "images"),
    }
    print("Импорт cam2 завершён:", summary)
    if skipped_no_image:
        print(f"  ВНИМАНИЕ: {skipped_no_image} .txt без парной картинки под {images_dir} "
              f"(укажите --images, если кадры лежат в другом месте).")
    print("Дальше: python -m vinery.training.yolo_dataset --task canopy --root dataset --group-by row")
    return summary


def _append_canopy_index_dedup(root: str | Path,
                               rows: list[tuple[str, str, str]]) -> None:
    """Дописать (name, bush_code, phase) в canopy_index.csv, создав заголовок при нужде."""
    if not rows:
        return
    path = canopy_src_dir(root) / CANOPY_INDEX_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "bush_code", "phase"])
        w.writerows(rows)


# ----------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ингест видео в датасеты: cam1 (vine_trunk) + стейджинг cam2 (canopy).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="видео прохода -> кадры в стейджинг на разметку")
    ex.add_argument("video")
    ex.add_argument("--vineyard", default="V1")
    ex.add_argument("--row", type=int, required=True)
    ex.add_argument("--pass", type=int, required=True, dest="pass_no")
    ex.add_argument("--staging", default=DEFAULT_STAGING)
    ex.add_argument("--fps", type=float, default=3.0,
                    help="целевая частота подвыборки кадров (по умолчанию 3)")
    ex.add_argument("--every", type=int, default=None,
                    help="брать каждый N-й кадр (приоритетнее --fps)")
    ex.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    ex.add_argument("--quality", type=int, default=90, dest="jpg_quality")

    si = sub.add_parser("stage-images", help="отдельные фото -> стейджинг с биркой ряда")
    si.add_argument("images_dir")
    si.add_argument("--vineyard", default="V1")
    si.add_argument("--row", type=int, required=True)
    si.add_argument("--staging", default=DEFAULT_STAGING)
    si.add_argument("--tag", default="photos")

    ce = sub.add_parser("canopy-extract",
                        help="cam2: видео -> кадры в стейджинг по фенофазе (canopy/<phase>)")
    ce.add_argument("video")
    ce.add_argument("--vineyard", default="V1")
    ce.add_argument("--row", type=int, required=True)
    ce.add_argument("--phase", required=True, choices=CANOPY_PHASES,
                    help="фенофаза партии (диктуется датой съёмки)")
    ce.add_argument("--pass", type=int, required=True, dest="pass_no")
    ce.add_argument("--staging", default=DEFAULT_CANOPY_STAGING)
    ce.add_argument("--fps", type=float, default=2.0,
                    help="целевая частота подвыборки кадров (по умолчанию 2)")
    ce.add_argument("--every", type=int, default=None,
                    help="брать каждый N-й кадр (приоритетнее --fps)")
    ce.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    ce.add_argument("--quality", type=int, default=90, dest="jpg_quality")

    cs = sub.add_parser("canopy-stage-images",
                        help="cam2: отдельные фото -> стейджинг фенофазы с биркой ряда")
    cs.add_argument("images_dir")
    cs.add_argument("--vineyard", default="V1")
    cs.add_argument("--row", type=int, required=True)
    cs.add_argument("--phase", required=True, choices=CANOPY_PHASES)
    cs.add_argument("--staging", default=DEFAULT_CANOPY_STAGING)

    im = sub.add_parser("import", help="cam1: YOLO-экспорт из CVAT -> раскладка задачи")
    im.add_argument("--labels", required=True, help="папка с YOLO .txt (obj_train_data)")
    im.add_argument("--images", default=None, help="папка с кадрами (по умолч. стейджинг)")
    im.add_argument("--root", default="dataset", dest="dataset_root")
    im.add_argument("--drop-empty", action="store_true",
                    help="не импортировать кадры без боксов (по умолч. оставляем как негативы)")

    ci = sub.add_parser("canopy-import",
                        help="cam2: YOLO-seg экспорт (полигоны) -> раскладка задачи canopy_seg")
    ci.add_argument("--labels", required=True, help="папка с YOLO-seg .txt (поиск рекурсивный)")
    ci.add_argument("--images", default=None,
                    help="папка с кадрами (по умолч. стейджинг canopy/, поиск рекурсивный)")
    ci.add_argument("--root", default="dataset", dest="dataset_root")
    ci.add_argument("--drop-empty", action="store_true",
                    help="не импортировать кадры без масок (по умолч. оставляем как негативы/healthy)")

    args = ap.parse_args()
    if args.cmd == "extract":
        extract_frames(args.video, vineyard=args.vineyard, row=args.row,
                       pass_no=args.pass_no, staging_root=args.staging,
                       fps=args.fps, every=args.every, max_frames=args.max_frames,
                       jpg_quality=args.jpg_quality)
    elif args.cmd == "stage-images":
        stage_images(args.images_dir, vineyard=args.vineyard, row=args.row,
                     staging_root=args.staging, tag=args.tag)
    elif args.cmd == "canopy-extract":
        extract_canopy_frames(args.video, vineyard=args.vineyard, row=args.row,
                              phase=args.phase, pass_no=args.pass_no,
                              staging_root=args.staging, fps=args.fps,
                              every=args.every, max_frames=args.max_frames,
                              jpg_quality=args.jpg_quality)
    elif args.cmd == "canopy-stage-images":
        stage_canopy_images(args.images_dir, vineyard=args.vineyard, row=args.row,
                            phase=args.phase, staging_root=args.staging)
    elif args.cmd == "import":
        import_labels(args.labels, images_dir=args.images,
                      dataset_root=args.dataset_root, keep_empty=not args.drop_empty)
    elif args.cmd == "canopy-import":
        import_canopy_labels(args.labels, images_dir=args.images,
                             dataset_root=args.dataset_root, keep_empty=not args.drop_empty)


if __name__ == "__main__":
    main()
