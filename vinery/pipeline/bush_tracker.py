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

Веса обучаются модулем vinery.training.train_trunk. Без model_path трекер
работает как «нет детекций» (None) — удобно для каркаса и тестов без torch.
"""
from __future__ import annotations

from typing import Optional

from .models import BushEvent, Frame

# Трекер для инференса (присвоение track_id). bytetrack — стабильный дефолт;
# при частых перекрытиях стволов попробуйте 'botsort.yaml'.
DEFAULT_TRACKER = "bytetrack.yaml"


class BushTracker:
    """Детектор и трекер ствола лозы (камера 1). Один ствол на куст -> 1:1 с кустом."""

    def __init__(self, model_path: Optional[str] = None, conf: float = 0.5,
                 tracker: str = DEFAULT_TRACKER):
        self.conf = conf
        self.tracker = tracker
        self.model = None
        if model_path:
            self._load(model_path)

    def _load(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "Не установлен ultralytics. Установите: pip install ultralytics") from e
        self.model = YOLO(model_path)   # веса детектора ствола (класс 'vine_trunk')

    def detect(self, frame: Frame) -> Optional[BushEvent]:
        """Вернуть BushEvent для ствола, мимо которого едем СЕЙЧАС (ближайшего к
        центру кадра), с устойчивым track_id. Если ствола нет — None.

        track_id ствола = личность куста в пределах прохода (лоза ↔ куст 1:1).
        Оценку болезни лозы (vine_health) добавит отдельная модель на кропе ствола.
        """
        if self.model is None:
            return None

        # persist=True — трекер держит состояние между кадрами потока.
        results = self.model.track(frame.image, persist=True, conf=self.conf,
                                   tracker=self.tracker, verbose=False)
        if not results:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return None

        # выбрать ствол, ближайший к центру кадра по горизонтали
        h, w = results[0].orig_shape  # (height, width)
        xywh = boxes.xywh.tolist()    # [cx, cy, bw, bh] в пикселях
        confs = boxes.conf.tolist()
        ids = boxes.id.int().tolist()
        center_x = w / 2.0
        best = min(range(len(xywh)), key=lambda i: abs(xywh[i][0] - center_x))

        cx, cy, bw, bh = xywh[best]
        return BushEvent(
            t=frame.t, track_id=int(ids[best]),
            bbox=(float(cx), float(cy), float(bw), float(bh)),
            confidence=float(confs[best]),
        )
