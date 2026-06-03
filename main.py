"""VineryMSM — потоковый сбор данных по ряду винограда.

Поток камеры 1 (куст) и камеры 2 (лист/соцветие/гроздь) идут параллельно.
Синхронизатор связывает их по track_id куста и пишет наблюдения в БД.

Запуск (когда подключите модели и источники видео):
    python main.py
"""
from __future__ import annotations

import time
from datetime import datetime

from vinery.db import Database
from vinery.dataset import build_dataset_skeleton
from vinery.pipeline import (
    VideoStream, merge_streams, BushTracker, CanopyAnalyzer, Synchronizer,
    ConstantSpeedLocalizer,
)


def run_pass(cam1_source, cam2_source, *,
             db_path="vinery.db", dataset_root="dataset",
             vineyard_code="V1", row_number=3, variety_code="cabernet_sauvignon",
             bush_model=None, canopy_models=None,
             bush_spacing_m=1.0, cart_speed_mps=0.5, localizer=None):
    """Один проход тележки вдоль ряда: читаем оба потока, пишем наблюдения.

    Идентификация куста привязана к физическому положению (localizer), а не к
    порядку детекций. По умолчанию — ConstantSpeedLocalizer (известная скорость
    проезда). Для поля подключите OdometryLocalizer/GpsLocalizer или метки на
    шпалере (BushEvent.marker_code).
    """
    build_dataset_skeleton(dataset_root)

    db = Database(db_path)
    vineyard_id = db.upsert_vineyard(vineyard_code)
    variety_id = db.upsert_variety(variety_code, variety_code)
    row_id = db.get_or_create_row(vineyard_id, variety_id, row_number,
                                  bush_spacing_m=bush_spacing_m)
    pass_id = db.create_pass(
        row_id, datetime.now().isoformat(timespec="seconds"),
        str(cam1_source), str(cam2_source),
    )

    if localizer is None:
        localizer = ConstantSpeedLocalizer(speed_mps=cart_speed_mps)

    tracker = BushTracker(model_path=bush_model)
    analyzer = CanopyAnalyzer(**(canopy_models or {}))
    sync = Synchronizer(db, pass_id, row_id, vineyard_code, row_number, variety_id,
                        localizer=localizer, bush_spacing_m=bush_spacing_m)

    cam1 = VideoStream(cam1_source, "cam1_bush")
    cam2 = VideoStream(cam2_source, "cam2_canopy")

    # Простейшая синхронизация двух потоков по времени кадра t.
    # (Для прод-версии лучше развести камеры по потокам/процессам и буферу.)
    g1, g2 = cam1.frames(), cam2.frames()
    f1 = next(g1, None)
    f2 = next(g2, None)
    while f1 is not None or f2 is not None:
        # обрабатываем тот кадр, у которого меньше временная метка
        take_cam1 = f2 is None or (f1 is not None and f1.t <= f2.t)

        if take_cam1:
            ev = tracker.detect(f1)
            if ev:
                sync.on_bush(ev)
            f1 = next(g1, None)
        else:
            res = analyzer.analyze(f2)
            sync.on_canopy(res)
            f2 = next(g2, None)

    sync.finish()
    cam1.release()
    cam2.release()
    db.close()
    print(f"Проход {pass_id} записан в {db_path}")


if __name__ == "__main__":
    # Подставьте свои источники: пути к файлам, RTSP-URL или индексы камер.
    # run_pass("cam1.mp4", "cam2.mp4")
    print("Заготовка готова. Подключите модели в bush_tracker.py / canopy_analyzer.py "
          "и источники видео в run_pass().")