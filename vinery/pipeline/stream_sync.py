"""Честное слияние двух видеопотоков в один поток, упорядоченный по ОБЩЕМУ времени.

Проблемы, которые это решает (vs наивное `f1.t <= f2.t` в main):
  - камеры физически разнесены вдоль ряда -> камера 2 видит тот же куст с
    задержкой/опережением относительно камеры 1 (cam2_offset_s);
  - у потоков может быть разный FPS и выпавшие кадры -> сравнивать надо по
    таймстемпам, а не по номерам кадров;
  - живым камерам нужны общие часы (см. VideoStream(realtime=..., start_monotonic=)).

Кадры камеры 2 сдвигаются на cam2_offset_s, чтобы «один и тот же куст» у обеих
камер совпадал во времени; затем оба потока сливаются в порядке общего времени.
"""
from __future__ import annotations

from typing import Iterable, Iterator

from .models import Frame

CAM1 = "cam1_bush"
CAM2 = "cam2_canopy"


def merge_streams(cam1: Iterable[Frame], cam2: Iterable[Frame],
                  cam2_offset_s: float = 0.0) -> Iterator[Frame]:
    """Слить два потока кадров, упорядочив по общему (выровненному) времени.

    cam2_offset_s — насколько ПОЗЖЕ камера 2 видит ту же точку ряда, что камера 1
    (положителен, если камера 2 смонтирована позади по ходу движения). Время
    кадров камеры 2 уменьшается на этот лаг для выравнивания с камерой 1.

    Кадры выдаются по неубыванию выровненного времени; при равенстве приоритет
    у камеры 1 (сначала «открыть/обновить куст», потом привязать к нему признаки).
    """
    it1, it2 = iter(cam1), iter(cam2)
    f1 = next(it1, None)
    f2 = next(it2, None)
    while f1 is not None or f2 is not None:
        t1 = f1.t if f1 is not None else float("inf")
        t2 = (f2.t - cam2_offset_s) if f2 is not None else float("inf")
        if t1 <= t2:
            yield f1
            f1 = next(it1, None)
        else:
            yield f2
            f2 = next(it2, None)
