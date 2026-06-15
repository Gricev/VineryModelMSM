"""Синхронизатор: связывает поток камеры 1 (личность куста) с потоком камеры 2
(признаки) в одну запись observation и пишет её в БД.

Логика «зависимости», о которой просил пользователь:
  - камера 1 трекает куст -> пока виден один и тот же track_id, наблюдение «открыто»;
  - все результаты камеры 2, пришедшие в это временное окно, агрегируются
    и привязываются к текущему кусту;
  - когда камера 1 переходит на следующий куст (track_id сменился) или поток
    закончился -> наблюдение «закрывается», агрегируется и пишется в БД.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

from .localization import Localizer
from .models import BushEvent, CanopyResult


@dataclass
class _Anchor:
    """Физическая привязка куста: слот вдоль ряда + измеренная координата."""
    slot: int                     # номер слота (position_in_row) — стабилен между проходами
    position_m: Optional[float]   # измеренное расстояние от начала ряда, м
    bush_code: str


@dataclass
class _OpenObservation:
    track_id: int
    anchor: _Anchor
    t_start: float
    t_end: float
    bush_frames: list = field(default_factory=list)   # кадры cam1
    canopy: list = field(default_factory=list)        # CanopyResult из cam2


class Synchronizer:
    """Склеивает два потока по track_id куста, но ИДЕНТИФИЦИРУЕТ куст по
    физическому якорю (метка на шпалере или координата от локализатора),
    а не по порядку детекций.

    db            — экземпляр Database
    pass_id       — id текущего прохода
    row_id        — id ряда (нужен для get_or_create_bush)
    vineyard_code, row_number — для генерации bush_code
    variety_id    — сорт ряда (один на ряд)
    localizer     — источник положения тележки (метры вдоль ряда); если None и
                    у событий нет marker_code, используется порядковый счётчик
                    (НЕ для продакшна — нумерация «сползёт» при пропуске куста).
    bush_spacing_m — шаг посадки, м: слот = round(position_m / bush_spacing_m).
    """

    def __init__(self, db, pass_id: int, row_id: int,
                 vineyard_code: str, row_number: int, variety_id: int,
                 localizer: Optional[Localizer] = None,
                 bush_spacing_m: float = 1.0):
        self.db = db
        self.pass_id = pass_id
        self.row_id = row_id
        self.vineyard_code = vineyard_code
        self.row_number = row_number
        self.variety_id = variety_id
        self.localizer = localizer
        self.bush_spacing_m = bush_spacing_m

        self._open: Optional[_OpenObservation] = None
        self._fallback_slot = 0                    # для запасного счётчика
        self._seen_tracks: dict[int, _Anchor] = {}  # track_id -> якорь (в пределах прохода)

    # --------------------------------------------------------- идентификация куста
    def _make_code(self, slot: int) -> str:
        return f"{self.vineyard_code}-R{self.row_number:02d}-B{slot:03d}"

    def _resolve_anchor(self, ev: BushEvent) -> _Anchor:
        """Определить физический слот куста для события камеры 1.

        Приоритет: 1) метка на шпалере (самый надёжный якорь);
                   2) координата от локализатора → слот по шагу посадки;
                   3) порядковый счётчик (запасной, ненадёжный).
        """
        if ev.marker_code:
            digits = re.findall(r"\d+", ev.marker_code)
            slot = int(digits[-1]) if digits else self._next_fallback()
            code = (ev.marker_code if "-" in ev.marker_code
                    else f"{self.vineyard_code}-R{self.row_number:02d}-{ev.marker_code}")
            return _Anchor(slot=slot, position_m=None, bush_code=code)

        if self.localizer is not None:
            pos = self.localizer.localize(ev.t).position_m
            slot = max(1, round(pos / self.bush_spacing_m) + 1)
            return _Anchor(slot=slot, position_m=pos, bush_code=self._make_code(slot))

        slot = self._next_fallback()
        return _Anchor(slot=slot, position_m=None, bush_code=self._make_code(slot))

    def _next_fallback(self) -> int:
        self._fallback_slot += 1
        return self._fallback_slot

    # --------------------------------------------------------- входящие события
    def on_bush(self, ev: BushEvent) -> None:
        """Камера 1 сообщает: в кадре куст с данным track_id."""
        if self._open is not None and self._continues_open(ev):
            self._open.t_end = ev.t
            self._open.bush_frames.append(ev)
            return

        # новый куст -> закрыть предыдущее наблюдение
        if self._open:
            self._flush()

        anchor = self._seen_tracks.get(ev.track_id)
        if anchor is None:
            anchor = self._resolve_anchor(ev)
            self._seen_tracks[ev.track_id] = anchor

        self._open = _OpenObservation(
            track_id=ev.track_id, anchor=anchor,
            t_start=ev.t, t_end=ev.t, bush_frames=[ev],
        )

    def _continues_open(self, ev: BushEvent) -> bool:
        """Продолжает ли событие ТЕКУЩЕЕ открытое наблюдение (тот же физический куст)?

        Тот же track_id -> всегда да (непрерывный трек = один куст; не дробим из-за
        дрожания слота на границе бина). Если track_id СМЕНИЛСЯ — это либо новый
        куст, либо трекер переоткрыл/задвоил тот же ствол под новым id; сливаем в
        текущий куст ТОЛЬКО если физический якорь (метка/координата) даёт тот же
        слот. Без якоря (fallback) смена track_id трактуется как новый куст —
        сохраняется прежнее поведение и инвариант test_anchoring.
        """
        open_ = self._open
        if ev.track_id == open_.track_id:
            return True
        slot = self._slot_only(ev)
        return slot is not None and slot == open_.anchor.slot

    def _slot_only(self, ev: BushEvent) -> Optional[int]:
        """Слот куста по физическому якорю БЕЗ побочных эффектов (не трогает
        запасной счётчик _next_fallback). Зеркалит приоритет _resolve_anchor
        (метка > координата). None — если якоря нет, слот определить нечем."""
        if ev.marker_code:
            digits = re.findall(r"\d+", ev.marker_code)
            return int(digits[-1]) if digits else None
        if self.localizer is not None:
            pos = self.localizer.localize(ev.t).position_m
            return max(1, round(pos / self.bush_spacing_m) + 1)
        return None

    def on_canopy(self, res: CanopyResult) -> None:
        """Камера 2 сообщает признаки. Привязываются к текущему открытому кусту."""
        if self._open is None:
            return  # камера 1 ещё не зафиксировала куст -> данные камеры 2 пропускаем
        self._open.canopy.append(res)

    def finish(self) -> None:
        """Вызвать в конце прохода, чтобы дописать последнее наблюдение."""
        if self._open:
            self._flush()

    # --------------------------------------------------------- агрегация + запись
    def _flush(self) -> None:
        obs = self._open
        self._open = None

        a = obs.anchor
        bush_id = self.db.get_or_create_bush(self.row_id, a.slot, a.bush_code, a.position_m)

        record = {
            "pass_id": self.pass_id,
            "bush_id": bush_id,
            "cam1_track_id": obs.track_id,
            "t_start": obs.t_start,
            "t_end": obs.t_end,
            "frames": [
                {"camera": "cam1_bush", "t": f.t, "path": "",
                 "bbox": list(f.bbox)} for f in obs.bush_frames
            ],
        }
        record.update(self._aggregate_vine(obs.bush_frames))   # лоза (cam1)
        record.update(self._aggregate_canopy(obs.canopy))      # лист/гроздь (cam2)
        self.db.write_observation(record)

    def _aggregate_vine(self, bush_frames: list) -> dict:
        """Свернуть оценки болезни лозы из кадров камеры 1 в одно решение по кусту."""
        items = [f.vine_health for f in bush_frames if f.vine_health]
        vine = self._vote(items, "disease_code")
        if not vine:
            return {}
        sev = [it["severity"] for it in items if it.get("severity") is not None]
        return {"vine_health": {
            "disease_code": vine["disease_code"],
            "severity": mean(sev) if sev else None,
            "confidence": vine["confidence"],
        }}

    def _aggregate_canopy(self, results: list[CanopyResult]) -> dict:
        """Свернуть множество кадров камеры 2 в одно решение по кусту.

        Стратегия: для классов — голос с максимальной средней уверенностью,
        для чисел (severity, ripeness) — среднее. Подстройте под свою задачу.
        """
        out: dict = {}

        variety = self._vote([r.variety for r in results if r.variety], "code")
        if variety:
            vid = self.db.upsert_variety(variety["code"], variety["code"])
            out["variety"] = {"variety_id": vid, "confidence": variety["confidence"]}

        leaf = self._vote([r.leaf_health for r in results if r.leaf_health], "disease_code")
        if leaf:
            sev = [r.leaf_health["severity"] for r in results
                   if r.leaf_health and r.leaf_health.get("severity") is not None]
            out["leaf_health"] = {
                "disease_code": leaf["disease_code"],
                "severity": mean(sev) if sev else None,
                "confidence": leaf["confidence"],
            }

        stage = self._vote([r.inflorescence_stage for r in results
                            if r.inflorescence_stage], "stage_code")
        if stage:
            out["inflorescence_stage"] = {
                "stage_code": stage["stage_code"], "confidence": stage["confidence"]}

        ripe = [r.cluster_ripeness for r in results if r.cluster_ripeness]
        if ripe:
            pcts = [c["ripeness_pct"] for c in ripe if c.get("ripeness_pct") is not None]
            brix = [c["brix_estimate"] for c in ripe if c.get("brix_estimate") is not None]
            confs = [c["confidence"] for c in ripe if c.get("confidence") is not None]
            out["cluster_ripeness"] = {
                "ripeness_pct": mean(pcts) if pcts else None,
                "brix_estimate": mean(brix) if brix else None,
                "confidence": mean(confs) if confs else None,
            }

        chealth = self._vote([r.cluster_health for r in results if r.cluster_health],
                             "disease_code")
        if chealth:
            sev = [r.cluster_health["severity"] for r in results
                   if r.cluster_health and r.cluster_health.get("severity") is not None]
            out["cluster_health"] = {
                "disease_code": chealth["disease_code"],
                "severity": mean(sev) if sev else None,
                "confidence": chealth["confidence"],
            }
        return out

    @staticmethod
    def _vote(items: list[dict], key: str) -> Optional[dict]:
        """Выбрать значение key с наибольшей суммарной уверенностью."""
        if not items:
            return None
        scores: dict[str, float] = {}
        for it in items:
            scores[it[key]] = scores.get(it[key], 0.0) + it.get("confidence", 1.0)
        best = max(scores, key=scores.get)
        confs = [it["confidence"] for it in items
                 if it[key] == best and it.get("confidence") is not None]
        return {key: best, "confidence": mean(confs) if confs else 1.0}