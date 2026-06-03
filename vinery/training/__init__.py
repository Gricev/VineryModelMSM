"""Обучение моделей VineryMSM.

Сейчас проработана задача камеры 1 — детекция (и трекинг) ствола лозы. Каркас
устроен так, чтобы запускаться поэтапно:

  synthetic   — сгенерировать синтетические кадры стволов + YOLO-разметку
                (работает на чистом Python + Pillow, без torch/ultralytics);
  yolo_dataset — собрать из размеченных кадров YOLO-датасет (train/val/test + data.yaml),
                 разрезая СТРОГО по кусту/ряду (split-by-bush, без утечки);
  train_trunk — обучить детектор лозы через Ultralytics (ленивый импорт; нужен
                установленный пакет `ultralytics`).

Идентификация куста в проде делается трекингом (`model.track`) — это инференс,
отдельного обучения трекер не требует (см. train_trunk и pipeline/bush_tracker).
"""
from .splits import Split, SplitRatios, assign_split, row_of, split_samples

__all__ = ["Split", "SplitRatios", "assign_split", "row_of", "split_samples"]