"""Загрузка кадров стейджинга в проект Roboflow через API (обход UI-лимитов).

Зачем: веб-загрузка Roboflow упирается в лимит выбора файлов и не принимает zip.
API заливает всю папку пачкой и СОХРАНЯЕТ имена файлов (в них бирки рядов
V1-R03/V1-R04) — поэтому после разметки и экспорта YOLO их подхватит обратный
приём со split-by-row. Модуль годится для обеих дорожек (различает по пути):
  - cam1 (`dataset/to_label/vine_trunk*`) -> проект Object Detection, боксы класса
    'vine_trunk', приём `ingest import`;
  - cam2 (`dataset/to_label/canopy/<phase>`) -> проект Instance Segmentation, МАСКИ
    органов/поражений по CANOPY_LABELS.md, приём `ingest canopy-import`.
Тип проекта (Detection vs Segmentation) задаётся при его создании в Roboflow —
заливка кадров одинаковая; следующий шаг печатается по пути стейджинга.

Подготовка:
    pip install roboflow
    API-ключ: app.roboflow.com -> Settings -> API Keys (Private API Key).
    workspace и project — из URL проекта: app.roboflow.com/<workspace>/<project>.

CLI:
    # cam1 (детекция ствола)
    python -m vinery.training.roboflow_upload ^
        --api-key XXXX --workspace my-ws --project vine-trunk ^
        --dir dataset/to_label/vine_trunk_light

    # cam2 (сегментация кроны: цветение)
    python -m vinery.training.roboflow_upload ^
        --api-key XXXX --workspace my-ws --project canopy-seg ^
        --dir dataset/to_label/canopy/flowering

Roboflow при экспорте может разложить кадры по своим train/valid/test —
это неважно: наш приём всё равно пересобирает сплит по рядам.
"""
from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
# Лог успешно залитых кадров внутри папки фото — основа возобновления.
UPLOADED_LOG = ".uploaded.txt"

def _load_uploaded(images_dir: Path) -> set[str]:
    """Имена уже залитых кадров из лога (для пропуска при возобновлении)."""
    log = images_dir / UPLOADED_LOG
    if not log.exists():
        return set()
    return {ln.strip() for ln in log.read_text("utf-8").splitlines() if ln.strip()}


def upload_dir(api_key: str, workspace: str, project: str, images_dir: str | Path,
               *, batch_name: str | None = "vinery-ingest",
               split: str | None = None, resume: bool = True) -> tuple[int, int]:
    """Залить все изображения из папки в проект Roboflow. Вернуть (успешно, ошибок).

    Возобновляемо: каждое успешное имя сразу дописывается в `.uploaded.txt`
    (с flush на диск), поэтому при вылете прогресс не теряется. При `resume=True`
    (по умолчанию) повторный запуск пропускает уже залитые кадры и доливает остаток.
    """
    try:
        from roboflow import Roboflow
    except ImportError as e:
        raise RuntimeError("Не установлен roboflow: pip install roboflow") from e

    images_dir = Path(images_dir)
    files = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"Нет изображений в {images_dir}")

    done = _load_uploaded(images_dir) if resume else set()
    todo = [f for f in files if f.name not in done]
    if done:
        print(f"Возобновление: уже залито {len(done)}, осталось {len(todo)} из {len(files)}.")
    if not todo:
        print("Все кадры уже залиты — нечего делать.")
        return 0, 0

    proj = Roboflow(api_key=api_key).workspace(workspace).project(project)
    print(f"Заливаю {len(todo)} кадров в {workspace}/{project} ...")
    ok = fail = 0
    log = images_dir / UPLOADED_LOG
    # Открыт на весь проход: append + flush после каждой удачи -> устойчиво к вылету.
    with log.open("a", encoding="utf-8") as logf:
        for i, f in enumerate(todo, 1):
            try:
                proj.upload(image_path=str(f), batch_name=batch_name,
                            split=split, num_retry_uploads=3)
                ok += 1
                logf.write(f.name + "\n")
                logf.flush()
            except Exception as e:                   # noqa: BLE001 — логируем и идём дальше
                fail += 1
                print(f"  [{i}/{len(todo)}] ОШИБКА {f.name}: {e}")
            if i % 20 == 0:
                print(f"  ... {i}/{len(todo)} (ok={ok}, fail={fail})")
    print(f"Готово: успешно {ok}, ошибок {fail} из {len(todo)}.")
    if fail:
        print(f"  {fail} кадров не залились — перезапустите ту же команду, "
              f"возобновление дольёт только их.")
    _print_next_steps(images_dir)
    return ok, fail


def _print_next_steps(images_dir: Path) -> None:
    """Подсказка по следующему шагу — зависит от дорожки (cam1 боксы / cam2 seg-маски).

    Различаем по пути стейджинга: 'canopy' в пути -> cam2 (сегментация, полигоны,
    canopy-import); иначе -> cam1 (детекция ствола, боксы, import).
    """
    if "canopy" in images_dir.parts:
        print(f"Дальше: проект Roboflow типа Instance Segmentation, разметьте МАСКАМИ "
              f"по CANOPY_LABELS.md (органы leaf/inflorescence + слой поражения), "
              f"Generate -> Export YOLOv8 (полигоны), затем:\n"
              f"  python -m vinery.training.ingest canopy-import "
              f"--labels <export>\\labels --images {images_dir}")
    else:
        print(f"Дальше: в Roboflow разметьте класс 'vine_trunk' (боксы), "
              f"Generate -> Export YOLOv8, затем:\n"
              f"  python -m vinery.training.ingest import "
              f"--labels <export>\\labels --images {images_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Загрузка кадров в Roboflow через API.")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--workspace", required=True, help="slug из URL проекта")
    ap.add_argument("--project", required=True, help="slug из URL проекта")
    ap.add_argument("--dir", default="dataset/to_label/vine_trunk_light",
                    dest="images_dir")
    ap.add_argument("--batch", default="vinery-ingest", dest="batch_name")
    ap.add_argument("--split", default=None, help="train/valid/test (необязательно)")
    ap.add_argument("--no-resume", action="store_false", dest="resume",
                    help="залить всё заново, игнорируя .uploaded.txt")
    args = ap.parse_args()
    upload_dir(args.api_key, args.workspace, args.project, args.images_dir,
               batch_name=args.batch_name, split=args.split, resume=args.resume)


if __name__ == "__main__":
    main()