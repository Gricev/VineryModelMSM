"""Камера 1: детекция и трекинг ЛОЗЫ (ствола) + оценка её болезни. Выдаёт BushEvent.

Камера-1 ищет не размытую «копну куста», а ствол лозы — дискретный вертикальный
объект, выходящий из земли в одной точке. Это детектится и трекается стабильнее,
а при формировке «один ствол на куст» лоза однозначно соответствует кусту (1:1):
найденная лоза = идентифицированный куст.

Две задачи на одном объекте:
  1) детекция + трекинг ствола лозы -> track_id (он же личность куста в ряду);
  2) классификация болезни той же лозы (ствол/рукав) -> vine_health.
Можно одной многоголовой моделью или двумя моделями на одном кадре.

Счёт лоз вдоль ряда даёт камере роль «линейки»: каждый новый ствол = следующий
куст (см. синхронизатор, резолв слота по track_id/marker_code/координате).

Сейчас это интерфейс-заглушка. Подключите модели в `detect()`:
например YOLOv8 + встроенный трекер (model.track(...)), который сразу даёт track_id.
"""
from __future__ import annotations

from typing import Optional

from .models import BushEvent, Frame


class BushTracker:
    """Детектор и трекер ствола лозы (камера 1). Один ствол на куст -> 1:1 с кустом."""

    def __init__(self, model_path: Optional[str] = None, conf: float = 0.5):
        self.conf = conf
        self.model = None
        if model_path:
            self._load(model_path)

    def _load(self, model_path: str) -> None:
        # from ultralytics import YOLO
        # self.model = YOLO(model_path)   # модель обучена на класс 'лоза/ствол'
        raise NotImplementedError(
            "Подключите модель детекции ЛОЗЫ (ствола) для камеры 1. "
            "Рекомендация: YOLOv8 с трекингом — model.track(frame, persist=True)."
        )

    def detect(self, frame: Frame) -> Optional[BushEvent]:
        """Вернуть BushEvent, если в кадре есть ствол лозы (с устойчивым track_id).

        track_id ствола = личность куста в пределах прохода (лоза ↔ куст 1:1).
        Если ствола в кадре нет — вернуть None.
        """
        if self.model is None:
            return None
        # results = self.model.track(frame.image, persist=True, conf=self.conf)
        # Выбрать ствол, пересекающий центр кадра (куст, мимо которого едем сейчас);
        # track_id = results...id ствола; bbox — рамка ствола.
        # vine = self.vine_model(frame.image)  # болезнь той же лозы по этому кадру
        # return BushEvent(t=frame.t, track_id=track_id, bbox=(x, y, w, h), confidence=...,
        #                  vine_health={'disease_code': 'esca', 'severity': 0.3, 'confidence': 0.8})
        raise NotImplementedError
