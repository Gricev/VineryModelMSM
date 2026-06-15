
"""Обучение сегментатора кроны для камеры 2 (Ultralytics YOLO-seg).

Что обучаем и что НЕ обучаем:
  - обучаем СЕГМЕНТАТОР (instance segmentation) по единому namespaced списку классов
    CANOPY_SEG_CLASSES (органы leaf/inflorescence/cluster + поражения leaf_*/cluster_*),
    см. CANOPY_LABELS.md. Один орган = один полигон; healthy органа = маска органа
    без слоя поражения (отдельного класса нет);
  - severity (доля поражённой площади) НЕ обучается — она считается из площадей масок
    в рантайме (CanopyAnalyzer -> CanopyResult), а не моделью;
  - сорт (variety), стадия соцветия (BBCH) и зрелость грозди (ripeness) — атрибуты/
    отдельные головы, не маски: в этот сегментатор не входят (см. CANOPY_LABELS.md §3).

Зеркало train_trunk.py, отличия: базовая модель *-seg.pt и сборка через
build_canopy_seg_dataset. ultralytics импортируется ЛЕНИВО — модуль и сборка
датасета работают без установленного torch.

Запуск (нужен установленный ultralytics; см. requirements.txt):
    python -m vinery.training.train_canopy --root dataset --epochs 50 --model yolov8n-seg.pt

Полный цикл от разметки:
    python -m vinery.training.ingest canopy-import --labels seg_export/labels
    python -m vinery.training.train_canopy --root dataset --build --epochs 1 --imgsz 320
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .splits import SplitRatios
from .yolo_dataset import build_canopy_seg_dataset


@dataclass
class CanopyTrainConfig:
    dataset_root: str = "dataset"
    yolo_dir: str = "dataset/yolo/canopy_seg"
    base_model: str = "yolov8n-seg.pt"  # *-seg.pt: n/s/m/l/x — компромисс скорость/точность
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str | int | None = None  # None -> авто (GPU при наличии), 'cpu', 0, '0,1'
    project: str = "runs/canopy_seg"
    name: str = "train"
    seed: int = 0


def train(cfg: CanopyTrainConfig, *, build: bool = False,
          group_by: str = "row", ratios: SplitRatios = SplitRatios()):
    """Обучить сегментатор кроны. build=True пересобирает YOLO-seg датасет перед обучением.

    Возвращает путь к лучшим весам (best.pt). Ultralytics импортируется ЛЕНИВО,
    чтобы модуль (и сборка датасета) работали без установленного torch.
    """
    if build:
        summary = build_canopy_seg_dataset(
            cfg.dataset_root, cfg.yolo_dir, group_by=group_by, ratios=ratios, seed=cfg.seed)
        print("Датасет собран:", summary["counts"], "| групп:", summary["groups"])

    data_yaml = Path(cfg.yolo_dir) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Нет {data_yaml}. Запустите с --build или сначала "
            f"vinery.training.yolo_dataset --task canopy.")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "Не установлен ultralytics. Установите: pip install ultralytics\n"
            "(подтянет torch). Подготовка датасета при этом уже выполнена — "
            "обучение можно запустить отдельно тем же модулем.") from e

    model = YOLO(cfg.base_model)   # *-seg.pt -> task=segment выводится автоматически
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=cfg.epochs, imgsz=cfg.imgsz, batch=cfg.batch,
        device=cfg.device, project=str(Path(cfg.project).resolve()),
        name=cfg.name, seed=cfg.seed,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Готово. Лучшие веса: {best}")
    print(f"Подключите их в CanopyAnalyzer(model_path='{best}'); "
          f"severity считается из масок инференса.")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение сегментатора кроны (cam2).")
    ap.add_argument("--root", default="dataset", dest="dataset_root")
    ap.add_argument("--yolo-dir", default="dataset/yolo/canopy_seg")
    ap.add_argument("--model", default="yolov8n-seg.pt", dest="base_model")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", action="store_true", help="пересобрать датасет перед обучением")
    ap.add_argument("--group-by", choices=["bush", "row"], default="row")
    args = ap.parse_args()

    cfg = CanopyTrainConfig(
        dataset_root=args.dataset_root, yolo_dir=args.yolo_dir, base_model=args.base_model,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, seed=args.seed)
    train(cfg, build=args.build, group_by=args.group_by)


if __name__ == "__main__":
    main()