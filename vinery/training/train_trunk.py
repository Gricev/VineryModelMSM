"""Обучение детектора ствола лозы для камеры 1 (Ultralytics YOLO).

Что обучаем и что НЕ обучаем:
  - обучаем ДЕТЕКТОР одного класса 'vine_trunk' (ствол лозы). Этого достаточно,
    чтобы в проде находить ствол в кадре;
  - трекинг (присвоение track_id = личность куста) — это ИНФЕРЕНС поверх детектора
    (`model.track(..., tracker='bytetrack.yaml')`), отдельная модель не обучается.
    Поэтому здесь только детекция; трекер подключается в pipeline/bush_tracker.py.
  - болезнь лозы (vine_health) — отдельная задача/модель на кропах ствола, не здесь.

Запуск (нужен установленный ultralytics; см. requirements.txt):
    python -m vinery.training.train_trunk --root dataset --epochs 50 --model yolov8n.pt

Полный smoke-цикл на синтетике:
    python -m vinery.training.synthetic --root dataset
    python -m vinery.training.train_trunk --root dataset --build --epochs 1 --imgsz 320
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .splits import SplitRatios
from .yolo_dataset import build_trunk_yolo_dataset

# Трекер по умолчанию для инференса (см. BushTracker). bytetrack стабилен и быстр;
# для частых перекрытий стволов можно переключить на 'botsort.yaml'.
DEFAULT_TRACKER = "bytetrack.yaml"


@dataclass
class TrunkTrainConfig:
    dataset_root: str = "dataset"
    yolo_dir: str = "dataset/yolo/vine_trunk"
    base_model: str = "yolov8n.pt"   # n/s/m/l/x — компромисс скорость/точность
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str | int | None = None  # None -> авто (GPU при наличии), 'cpu', 0, '0,1'
    project: str = "runs/vine_trunk"
    name: str = "train"
    seed: int = 0


def train(cfg: TrunkTrainConfig, *, build: bool = False,
          group_by: str = "row", ratios: SplitRatios = SplitRatios()):
    """Обучить детектор лозы. build=True пересобирает YOLO-датасет перед обучением.

    Возвращает путь к лучшим весам (best.pt). Ultralytics импортируется ЛЕНИВО,
    чтобы модуль (и сборка датасета) работали без установленного torch.
    """
    if build:
        summary = build_trunk_yolo_dataset(
            cfg.dataset_root, cfg.yolo_dir, group_by=group_by, ratios=ratios, seed=cfg.seed)
        print("Датасет собран:", summary["counts"], "| групп:", summary["groups"])

    data_yaml = Path(cfg.yolo_dir) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Нет {data_yaml}. Запустите с --build или сначала vinery.training.yolo_dataset.")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError(
            "Не установлен ultralytics. Установите: pip install ultralytics\n"
            "(подтянет torch). Подготовка датасета при этом уже выполнена — "
            "обучение можно запустить отдельно тем же модулем.") from e

    model = YOLO(cfg.base_model)
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=cfg.epochs, imgsz=cfg.imgsz, batch=cfg.batch,
        device=cfg.device, project=cfg.project, name=cfg.name, seed=cfg.seed,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Готово. Лучшие веса: {best}")
    print(f"Подключите их в BushTracker(model_path='{best}') — трекинг: {DEFAULT_TRACKER}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение детектора ствола лозы (cam1).")
    ap.add_argument("--root", default="dataset", dest="dataset_root")
    ap.add_argument("--yolo-dir", default="dataset/yolo/vine_trunk")
    ap.add_argument("--model", default="yolov8n.pt", dest="base_model")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", action="store_true", help="пересобрать датасет перед обучением")
    ap.add_argument("--group-by", choices=["bush", "row"], default="row")
    args = ap.parse_args()

    cfg = TrunkTrainConfig(
        dataset_root=args.dataset_root, yolo_dir=args.yolo_dir, base_model=args.base_model,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, seed=args.seed)
    train(cfg, build=args.build, group_by=args.group_by)


if __name__ == "__main__":
    main()