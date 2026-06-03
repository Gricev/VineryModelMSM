"""Структуры данных, которыми обмениваются части pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Frame:
    """Один кадр из потока."""
    camera: str            # 'cam1_bush' | 'cam2_canopy'
    t: float               # секунды от старта прохода
    index: int             # порядковый номер кадра в потоке
    image: "any"           # numpy-массив (BGR); тип не импортируем, чтобы не тянуть cv2 сюда


@dataclass
class BushEvent:
    """Результат камеры 1: в кадре обнаружена ЛОЗА (ствол).

    Камера 1 детектит ствол лозы (а не размытую копну куста) и трекает его.
    При формировке «один ствол на куст» лоза ↔ куст 1:1, поэтому track_id ствола
    служит личностью куста. На той же лозе оценивается её здоровье (vine_health:
    ствол / рукав — эска, эутипоз и т.п.) — привязывается к кусту напрямую.
    """
    t: float
    track_id: int                  # id трека ЛОЗЫ = личность куста (только в пределах прохода)
    bbox: tuple[float, float, float, float]  # рамка ствола: x, y, w, h
    confidence: float
    vine_health: Optional[dict] = None  # {'disease_code','severity','confidence'} лоза
    marker_code: Optional[str] = None   # код физической метки куста (QR/ArUco/RFID), если есть


@dataclass
class CanopyResult:
    """Результат камеры 2 по кадру: признаки куста.

    Любое поле может быть None, если соответствующий объект ещё не виден
    (например, грозди нет в начале сезона -> cluster_ripeness=None).
    """
    t: float
    variety: Optional[dict] = None              # {'code','confidence'}
    leaf_health: Optional[dict] = None          # {'disease_code','severity','confidence'}
    inflorescence_stage: Optional[dict] = None  # {'stage_code','confidence'}
    cluster_ripeness: Optional[dict] = None      # {'ripeness_pct','brix_estimate','confidence'}
    cluster_health: Optional[dict] = None        # {'disease_code','severity','confidence'} болезнь грозди
    frame_path: Optional[str] = None
