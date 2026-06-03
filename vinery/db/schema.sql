-- Схема БД VineryMSM.
-- Иерархия: vineyard -> vine_row -> bush -> pass -> observation -> {результаты моделей}
-- Связующая сущность — bush. Камера 1 даёт track_id куста, камера 2 — признаки.

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------------ справочники
CREATE TABLE IF NOT EXISTS variety (        -- сорт винограда
    id    INTEGER PRIMARY KEY,
    code  TEXT UNIQUE NOT NULL,             -- 'cabernet_sauvignon'
    name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vineyard (       -- виноградник / участок
    id        INTEGER PRIMARY KEY,
    code      TEXT UNIQUE NOT NULL,         -- 'V1'
    name      TEXT,
    latitude  REAL,
    longitude REAL
);

-- ------------------------------------------------------------------ иерархия
CREATE TABLE IF NOT EXISTS vine_row (       -- ряд (один сорт на ряд)
    id            INTEGER PRIMARY KEY,
    vineyard_id   INTEGER NOT NULL REFERENCES vineyard(id),
    variety_id    INTEGER NOT NULL REFERENCES variety(id),
    row_number    INTEGER NOT NULL,
    bush_spacing_m REAL DEFAULT 1.0,        -- шаг посадки кустов вдоль ряда, м
    start_offset_m REAL DEFAULT 0.0,        -- координата первого куста от начала ряда
    UNIQUE (vineyard_id, row_number)
);

CREATE TABLE IF NOT EXISTS bush (           -- куст
    id              INTEGER PRIMARY KEY,
    row_id          INTEGER NOT NULL REFERENCES vine_row(id),
    position_in_row INTEGER NOT NULL,       -- номер СЛОТА вдоль ряда (физический якорь)
    position_m      REAL,                   -- измеренное расстояние от начала ряда, м
    bush_code       TEXT UNIQUE NOT NULL,   -- 'V1-R03-B017'
    UNIQUE (row_id, position_in_row)
);

CREATE TABLE IF NOT EXISTS pass (           -- проход / сессия съёмки
    id           INTEGER PRIMARY KEY,
    row_id       INTEGER NOT NULL REFERENCES vine_row(id),
    started_at   TEXT NOT NULL,             -- ISO datetime
    ended_at     TEXT,
    direction    TEXT,                      -- 'forward' / 'backward'
    cam1_source  TEXT,                      -- путь/URL потока камеры куста
    cam2_source  TEXT,                      -- путь/URL потока камеры листа/грозди
    notes        TEXT
);

-- --------------------------------------------------- синхронизированное наблюдение
-- Одна строка = один куст в рамках одного прохода (cam1 + cam2 склеены по времени).
CREATE TABLE IF NOT EXISTS observation (
    id            INTEGER PRIMARY KEY,
    pass_id       INTEGER NOT NULL REFERENCES pass(id),
    bush_id       INTEGER NOT NULL REFERENCES bush(id),
    cam1_track_id INTEGER,                  -- id трека куста на камере 1
    t_start       REAL NOT NULL,            -- сек от старта прохода (вход в куст)
    t_end         REAL,
    created_at    TEXT DEFAULT (datetime('now')),
    UNIQUE (pass_id, bush_id, t_start)
);

CREATE TABLE IF NOT EXISTS frame (          -- кадр, привязанный к наблюдению
    id             INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES observation(id),
    camera         TEXT NOT NULL,           -- 'cam1_bush' | 'cam2_canopy'
    t              REAL NOT NULL,           -- timestamp кадра (сек от старта)
    path           TEXT NOT NULL,
    bbox           TEXT                     -- json [x,y,w,h] куста (для cam1)
);

-- ------------------------------------------------- результаты моделей (агрегаты)
CREATE TABLE IF NOT EXISTS variety_prediction (
    id             INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES observation(id),
    variety_id     INTEGER REFERENCES variety(id),
    confidence     REAL
);

CREATE TABLE IF NOT EXISTS leaf_health (    -- болезнь ЛИСТА (камера 2)
    id             INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES observation(id),
    disease_code   TEXT NOT NULL,           -- 'healthy','mildew','oidium','black_rot'
    severity       REAL,                    -- 0..1 доля поражённой площади
    confidence     REAL,
    frame_id       INTEGER REFERENCES frame(id)
);

CREATE TABLE IF NOT EXISTS vine_health (    -- болезнь ЛОЗЫ (камера 1: ствол/рукав/побег)
    id             INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES observation(id),
    disease_code   TEXT NOT NULL,           -- 'healthy','esca','eutypa','botryosphaeria'
    severity       REAL,                    -- 0..1
    confidence     REAL,
    frame_id       INTEGER REFERENCES frame(id)
);

CREATE TABLE IF NOT EXISTS inflorescence_stage (
    id             INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES observation(id),
    stage_code     TEXT NOT NULL,           -- BBCH-подобная стадия, напр. 'BBCH-65'
    confidence     REAL,
    frame_id       INTEGER REFERENCES frame(id)
);

CREATE TABLE IF NOT EXISTS cluster_ripeness (   -- зрелость ГРОЗДИ (камера 2, сезонно)
    id             INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES observation(id),
    ripeness_pct   REAL,                    -- 0..100 (появляется, когда есть гроздь)
    brix_estimate  REAL,
    confidence     REAL,
    frame_id       INTEGER REFERENCES frame(id)
);

CREATE TABLE IF NOT EXISTS cluster_health (    -- болезнь ГРОЗДИ (камера 2, сезонно)
    id             INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES observation(id),
    disease_code   TEXT NOT NULL,           -- 'healthy','botrytis','sour_rot','berry_oidium'
    severity       REAL,                    -- 0..1 доля поражённых ягод
    confidence     REAL,
    frame_id       INTEGER REFERENCES frame(id)
);

CREATE INDEX IF NOT EXISTS ix_observation_bush ON observation(bush_id);
CREATE INDEX IF NOT EXISTS ix_observation_pass ON observation(pass_id);
CREATE INDEX IF NOT EXISTS ix_frame_obs        ON frame(observation_id);
