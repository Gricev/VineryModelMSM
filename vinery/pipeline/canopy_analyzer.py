"""Камера 2: сорт, болезнь листа, стадия соцветия, % созревания грозди.

Заглушка-интерфейс. Подключите 4 модели (или одну многоголовую) в `analyze()`.
Каждое поле независимо: пока грозди нет — cluster_ripeness остаётся None.
"""
from __future__ import annotations

from typing import Optional

from .models import CanopyResult, Frame


class CanopyAnalyzer:
    def __init__(self,
                 variety_model: Optional[str] = None,
                 leaf_model: Optional[str] = None,
                 inflorescence_model: Optional[str] = None,
                 cluster_model: Optional[str] = None):
        self.variety_model = None
        self.leaf_model = None
        self.inflorescence_model = None
        self.cluster_model = None
        # TODO: загрузить модели по путям (YOLO/классификаторы/регрессор).

    def analyze(self, frame: Frame) -> CanopyResult:
        """Прогнать модели по кадру камеры 2 и собрать признаки."""
        result = CanopyResult(t=frame.t)
        # result.variety = {'code': 'cabernet_sauvignon', 'confidence': 0.97}
        # result.leaf_health = {'disease_code': 'mildew', 'severity': 0.18, 'confidence': 0.88}
        # result.inflorescence_stage = {'stage_code': 'BBCH-65', 'confidence': 0.81}
        # result.cluster_ripeness = {'ripeness_pct': 42.0, 'brix_estimate': 14.5, 'confidence': 0.76}
        # result.cluster_health = {'disease_code': 'botrytis', 'severity': 0.12, 'confidence': 0.79}
        return result