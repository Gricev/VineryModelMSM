"""Слой доступа к БД (SQLite). Создаёт схему и пишет синхронизированные наблюдения."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.commit()

    # ---------------------------------------------------------- справочники
    def upsert_variety(self, code: str, name: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO variety(code, name) VALUES(?, ?) "
            "ON CONFLICT(code) DO UPDATE SET name=excluded.name RETURNING id",
            (code, name),
        )
        return cur.fetchone()[0]

    def upsert_vineyard(self, code: str, name: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO vineyard(code, name) VALUES(?, ?) "
            "ON CONFLICT(code) DO UPDATE SET name=excluded.name RETURNING id",
            (code, name),
        )
        return cur.fetchone()[0]

    def get_or_create_row(self, vineyard_id: int, variety_id: int, row_number: int,
                          bush_spacing_m: float = 1.0, start_offset_m: float = 0.0) -> int:
        row = self.conn.execute(
            "SELECT id FROM vine_row WHERE vineyard_id=? AND row_number=?",
            (vineyard_id, row_number),
        ).fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO vine_row(vineyard_id, variety_id, row_number, "
            "bush_spacing_m, start_offset_m) VALUES(?,?,?,?,?)",
            (vineyard_id, variety_id, row_number, bush_spacing_m, start_offset_m),
        )
        return cur.lastrowid

    def get_or_create_bush(self, row_id: int, position: int, bush_code: str,
                           position_m: float | None = None) -> int:
        """Резолв куста по СЛОТУ (position_in_row) внутри ряда — стабильный физический
        якорь. Один и тот же слот в разных проходах → один и тот же куст (id/bush_code)."""
        row = self.conn.execute(
            "SELECT id FROM bush WHERE row_id=? AND position_in_row=?",
            (row_id, position),
        ).fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO bush(row_id, position_in_row, position_m, bush_code) "
            "VALUES(?,?,?,?)",
            (row_id, position, position_m, bush_code),
        )
        return cur.lastrowid

    def create_pass(self, row_id: int, started_at: str,
                    cam1_source: str, cam2_source: str,
                    direction: str = "forward") -> int:
        cur = self.conn.execute(
            "INSERT INTO pass(row_id, started_at, direction, cam1_source, cam2_source) "
            "VALUES(?,?,?,?,?)",
            (row_id, started_at, direction, cam1_source, cam2_source),
        )
        self.conn.commit()
        return cur.lastrowid

    # ------------------------------------------------- запись наблюдения целиком
    def write_observation(self, record: dict) -> int:
        """Пишет одно синхронизированное наблюдение (см. synchronizer.ObservationRecord).

        record = {
          pass_id, bush_id, cam1_track_id, t_start, t_end,
          frames: [{camera, t, path, bbox}],
          variety: {variety_id, confidence} | None,
          leaf_health: {disease_code, severity, confidence} | None,
          inflorescence_stage: {stage_code, confidence} | None,
          cluster_ripeness: {ripeness_pct, brix_estimate, confidence} | None,
        }
        """
        cur = self.conn.execute(
            "INSERT INTO observation(pass_id, bush_id, cam1_track_id, t_start, t_end) "
            "VALUES(?,?,?,?,?)",
            (record["pass_id"], record["bush_id"], record.get("cam1_track_id"),
             record["t_start"], record.get("t_end")),
        )
        obs_id = cur.lastrowid

        for fr in record.get("frames", []):
            self.conn.execute(
                "INSERT INTO frame(observation_id, camera, t, path, bbox) VALUES(?,?,?,?,?)",
                (obs_id, fr["camera"], fr["t"], fr["path"],
                 json.dumps(fr.get("bbox")) if fr.get("bbox") else None),
            )

        if v := record.get("variety"):
            self.conn.execute(
                "INSERT INTO variety_prediction(observation_id, variety_id, confidence) "
                "VALUES(?,?,?)",
                (obs_id, v.get("variety_id"), v.get("confidence")),
            )
        if lh := record.get("leaf_health"):
            self.conn.execute(
                "INSERT INTO leaf_health(observation_id, disease_code, severity, confidence) "
                "VALUES(?,?,?,?)",
                (obs_id, lh["disease_code"], lh.get("severity"), lh.get("confidence")),
            )
        if vh := record.get("vine_health"):
            self.conn.execute(
                "INSERT INTO vine_health(observation_id, disease_code, severity, confidence) "
                "VALUES(?,?,?,?)",
                (obs_id, vh["disease_code"], vh.get("severity"), vh.get("confidence")),
            )
        if ist := record.get("inflorescence_stage"):
            self.conn.execute(
                "INSERT INTO inflorescence_stage(observation_id, stage_code, confidence) "
                "VALUES(?,?,?)",
                (obs_id, ist["stage_code"], ist.get("confidence")),
            )
        if cr := record.get("cluster_ripeness"):
            self.conn.execute(
                "INSERT INTO cluster_ripeness(observation_id, ripeness_pct, brix_estimate, confidence) "
                "VALUES(?,?,?,?)",
                (obs_id, cr.get("ripeness_pct"), cr.get("brix_estimate"), cr.get("confidence")),
            )
        if ch := record.get("cluster_health"):
            self.conn.execute(
                "INSERT INTO cluster_health(observation_id, disease_code, severity, confidence) "
                "VALUES(?,?,?,?)",
                (obs_id, ch["disease_code"], ch.get("severity"), ch.get("confidence")),
            )

        self.conn.commit()
        return obs_id

    def close(self) -> None:
        self.conn.close()