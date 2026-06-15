"""Обучение моделей VineryMSM.

Сейчас проработана задача камеры 1 — детекция (и трекинг) ствола лозы. Каркас
устроен так, чтобы запускаться поэтапно:

  synthetic   — сгенерировать синтетические кадры стволов + YOLO-разметку
                (работает на чистом Python + Pillow, без torch/ultralytics);
  ingest      — РЕАЛЬНЫЕ видео проездов -> кадры на разметку (extract) и приём
                YOLO-экспорта из CVAT/Roboflow обратно в датасет (import);
  sam2_prelabel — авто-предразметка стволов через SAM2: маска -> YOLO-бокс
                (черновик под ручную проверку; для фото — режим images);
  yolo_dataset — собрать из размеченных кадров YOLO-датасет (train/val/test + data.yaml),
                 разрезая СТРОГО по кусту/ряду (split-by-bush, без утечки);
                 --task trunk (боксы, cam1) | --task canopy (полигоны YOLO-seg, cam2);
  train_trunk — обучить детектор лозы через Ultralytics (ленивый импорт; нужен
                установленный пакет `ultralytics`).

Идентификация куста в проде делается трекингом (`model.track`) — это инференс,
отдельного обучения трекер не требует (см. train_trunk и pipeline/bush_tracker).

Камера 2 (крона: лист/соцветие/гроздь) — отдельная дорожка с той же раскладкой и
анти-утечкой по ряду, но СЕГМЕНТАЦИЯ (маски, не боксы); таксономия — CANOPY_LABELS.md:

  ingest canopy-extract / canopy-stage-images — кадры в стейджинг по фенофазе;
  sam2_canopy — авто-ПРЕДразметка слоя органа: маска SAM2 -> ПОЛИГОН YOLO-seg
                (черновик; класс органа из фенофазы; auto-режим для множества листьев);
  ingest canopy-import — приём YOLO-seg экспорта (полигоны) -> annotations/canopy_seg;
  yolo_dataset --task canopy — сборка YOLO-seg датасета (split по ряду);
  train_canopy — обучить сегментатор кроны (Ultralytics *-seg.pt; ленивый импорт).

severity (доля поражённой площади) НЕ обучается и НЕ пишется при приёме — считается
из площадей масок в рантайме (CanopyAnalyzer -> CanopyResult).
"""
from .splits import Split, SplitRatios, assign_split, row_of, split_samples

__all__ = ["Split", "SplitRatios", "assign_split", "row_of", "split_samples"]