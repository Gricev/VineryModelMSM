"""Локализация тележки вдоль ряда — физический якорь для идентификации куста.

Зачем: без внешней привязки к местности номер куста приходится считать
порядковым счётчиком детекций. Стоит камере пропустить или задвоить куст —
и вся последующая нумерация «сползает», данные пишутся не тому кусту.
Локализатор даёт положение (метры от начала ряда) в любой момент времени,
из которого детерминированно вычисляется СЛОТ куста = round(position / spacing).
Один и тот же физический слот → один и тот же bush_code в любом проходе.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Localization:
    position_m: float            # расстояние вдоль ряда от его начала, метры
    lat: Optional[float] = None
    lon: Optional[float] = None


class Localizer:
    """Интерфейс источника положения. Реализуйте localize() под свой датчик."""

    def localize(self, t: float) -> Localization:
        raise NotImplementedError


class ConstantSpeedLocalizer(Localizer):
    """Постоянная скорость тележки. Годится для калибровки и проигрывания
    записанных файлов, где скорость проезда известна и стабильна."""

    def __init__(self, speed_mps: float, start_offset_m: float = 0.0):
        self.speed = speed_mps
        self.start = start_offset_m

    def localize(self, t: float) -> Localization:
        return Localization(position_m=self.start + self.speed * t)


class OdometryLocalizer(Localizer):
    """Скелет: положение по одометрии (энкодер колеса/гусеницы).

    Подключите чтение тиков энкодера с таймстемпами и пересчёт тик→метры.
    Точнее постоянной скорости, устойчив к остановкам и неравномерному ходу.
    """

    def __init__(self, ticks_per_meter: float):
        self.ticks_per_meter = ticks_per_meter
        # TODO: буфер (t, ticks) из драйвера энкодера

    def localize(self, t: float) -> Localization:
        raise NotImplementedError(
            "Подключите поток одометрии: интерполяция позиции по (t, ticks).")


class GpsLocalizer(Localizer):
    """Скелет: положение по GPS/RTK, спроецированное на осевую линию ряда.

    Нужны: поток координат (t, lat, lon) и геометрия ряда (начало + азимут),
    чтобы спроецировать точку на ось ряда и получить position_m.
    """

    def __init__(self, row_start: tuple[float, float], row_azimuth_deg: float):
        self.row_start = row_start
        self.row_azimuth_deg = row_azimuth_deg

    def localize(self, t: float) -> Localization:
        raise NotImplementedError(
            "Подключите поток GPS/RTK и проекцию координаты на ось ряда.")