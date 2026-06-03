"""Проверка физического якорения куста (пункт 1).

Ключевое свойство: bush_code определяется ПОЛОЖЕНИЕМ куста, а не порядком
детекций. Поэтому пропуск куста в одном из проходов не сдвигает нумерацию
остальных, и один и тот же физический куст получает один и тот же код в любом
проходе.
"""
import os
import tempfile

from vinery.db import Database
from vinery.pipeline import BushEvent, Synchronizer, ConstantSpeedLocalizer


def _run_pass(db, row_id, variety_id, sightings, spacing=1.0, speed=0.5):
    """sightings: список (track_id, t) — когда камера 1 видела куст."""
    pass_id = db.create_pass(row_id, "2026-06-03T10:00", "c1", "c2")
    loc = ConstantSpeedLocalizer(speed_mps=speed)
    sync = Synchronizer(db, pass_id, row_id, "V1", 3, variety_id,
                        localizer=loc, bush_spacing_m=spacing)
    for track_id, t in sightings:
        sync.on_bush(BushEvent(t=t, track_id=track_id, bbox=(0, 0, 1, 1), confidence=0.9))
    sync.finish()
    return pass_id


def _codes(db, pass_id):
    rows = db.conn.execute(
        "SELECT b.bush_code FROM observation o JOIN bush b ON b.id=o.bush_id "
        "WHERE o.pass_id=? ORDER BY o.t_start", (pass_id,)
    ).fetchall()
    return [r["bush_code"] for r in rows]


def test_skipped_bush_does_not_shift_numbering():
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    try:
        db = Database(path)
        vid = db.upsert_vineyard("V1"); var = db.upsert_variety("cs", "CS")
        row = db.get_or_create_row(vid, var, 3, bush_spacing_m=1.0)

        # Проход 1: 3 куста на позициях ~1, 2, 3 м (скорость 0.5 м/с -> t=2,4,6).
        p1 = _run_pass(db, row, var, [(10, 2.0), (11, 4.0), (12, 6.0)])
        codes1 = _codes(db, p1)

        # Проход 2: ДРУГИЕ track_id, средний куст ПРОПУЩЕН камерой.
        p2 = _run_pass(db, row, var, [(20, 2.0), (22, 6.0)])
        codes2 = _codes(db, p2)

        # Слот = round(pos/1.0)+1: pos1->2, pos2->3, pos3->4
        assert codes1 == ["V1-R03-B002", "V1-R03-B003", "V1-R03-B004"]
        # Пропуск среднего НЕ сдвинул третий куст на B003 — он остался B004.
        assert codes2 == ["V1-R03-B002", "V1-R03-B004"]

        # Один физический куст -> один и тот же bush_id в обоих проходах.
        n_bushes = db.conn.execute("SELECT COUNT(*) c FROM bush").fetchone()["c"]
        assert n_bushes == 3   # не 5: проходы переиспользовали те же кусты
        db.close()
    finally:
        os.remove(path)


def test_marker_code_takes_priority():
    fd, path = tempfile.mkstemp(suffix=".db"); os.close(fd)
    try:
        db = Database(path)
        vid = db.upsert_vineyard("V1"); var = db.upsert_variety("cs", "CS")
        row = db.get_or_create_row(vid, var, 3)
        pass_id = db.create_pass(row, "2026-06-03T10:00", "c1", "c2")
        sync = Synchronizer(db, pass_id, row, "V1", 3, var,
                            localizer=ConstantSpeedLocalizer(0.5), bush_spacing_m=1.0)
        # Метка на шпалере имеет приоритет над координатой.
        sync.on_bush(BushEvent(t=2.0, track_id=1, bbox=(0, 0, 1, 1),
                               confidence=0.9, marker_code="B042"))
        sync.finish()
        assert _codes(db, pass_id) == ["V1-R03-B042"]
        db.close()
    finally:
        os.remove(path)