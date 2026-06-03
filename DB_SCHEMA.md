# Схема базы данных VineryMSM

Описание собрано из живой схемы [`vinery/db/schema.sql`](vinery/db/schema.sql);
запись выполняет [`vinery/db/database.py`](vinery/db/database.py). Всё, что в
примере ниже, реально пишется кодом и проверено прогонами.

## Карта таблиц (от общего к частному)

```
┌──────────────┐        ┌──────────────┐
│   vineyard   │        │   variety    │   ← справочники
│  V1, "Юг-1"  │        │ cabernet_s.. │
└──────┬───────┘        └──────┬───────┘
       │                        │
       │  ┌─────────────────────┘
       ▼  ▼
   ┌──────────────┐
   │   vine_row   │   ряд = 1 сорт
   │  V1, ряд 3   │
   └──────┬───────┘
          │
     ┌────┴─────┐
     ▼          ▼
┌─────────┐  ┌─────────┐
│  bush   │  │  pass   │   куст (где) │ проход (когда)
│V1-R03-  │  │ 3 июня  │
│ B017    │  │ 10:15   │
└────┬────┘  └────┬────┘
     │            │
     └─────┬──────┘
           ▼
   ┌────────────────┐
   │  observation   │  ★ один куст в одном проходе (cam1+cam2 склеены)
   │  bush=17,pass=1│
   └───┬────────────┘
       │  один-ко-многим / один-к-одному
       ├──────────┬──────────┬──────────────┬──────────┬──────────────┬──────────────┐
       ▼          ▼          ▼              ▼          ▼              ▼              ▼
  ┌────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐ ┌────────────┐
  │ frame  │ │vine_     │ │leaf_     │ │variety_    │ │inflores- │ │cluster_    │ │cluster_    │
  │ кадры  │ │health    │ │health    │ │prediction  │ │cence_    │ │ripeness    │ │health      │
  │cam1+2  │ │ЛОЗА cam1 │ │ЛИСТ cam2 │ │сорт  cam2  │ │stage cam2│ │зрелость c2 │ │болезнь c2  │
  └────────┘ └──────────┘ └──────────┘ └────────────┘ └──────────┘ └────────────┘ └────────────┘
```

## Где какая таблица применяется

| Слой | Таблицы | Кто пишет | Когда |
|---|---|---|---|
| **Справочники** | `vineyard`, `variety` | настройка системы | один раз (`upsert_*`) |
| **Топология поля** | `vine_row`, `bush` | при первом проходе | `get_or_create_*` |
| **Сессия съёмки** | `pass` | старт прохода | `create_pass` в начале |
| **Связка камер** ★ | `observation` | синхронизатор | когда камера 1 уходит с куста (`_flush`) |
| **Сырьё** | `frame` | синхронизатор | вместе с observation |
| **Болезни/признаки** | `vine_health`, `leaf_health`, `variety_prediction`, `inflorescence_stage`, `cluster_ripeness`, `cluster_health` | синхронизатор (агрегаты) | вместе с observation |

## Таблицы по полям

### Справочники
- **`variety`** — сорт: `id`, `code` (уникальный, напр. `cabernet_sauvignon`), `name`.
- **`vineyard`** — участок: `id`, `code` (`V1`), `name`, `latitude`, `longitude`.

### Иерархия
- **`vine_row`** — ряд (один сорт): `id`, `vineyard_id→vineyard`, `variety_id→variety`,
  `row_number`, `bush_spacing_m` (шаг посадки), `start_offset_m`.
  Уникум: `(vineyard_id, row_number)`.
- **`bush`** — куст: `id`, `row_id→vine_row`, `position_in_row` (номер СЛОТА —
  физический якорь), `position_m` (измеренное расстояние от начала ряда),
  `bush_code` (уникальный, `V1-R03-B017`). Уникум: `(row_id, position_in_row)`.
  Резолв идёт по `(row_id, position_in_row)`: один и тот же слот в любом проходе →
  один и тот же куст. Слот = `round(position_m / bush_spacing_m)`, где `position_m`
  даёт локализатор (одометрия/GPS) или метка на шпалере. Нумерация привязана к
  положению, а не к порядку детекций — пропуск куста не сдвигает остальные.
- **`pass`** — проход/сессия: `id`, `row_id→vine_row`, `started_at`, `ended_at`,
  `direction`, `cam1_source`, `cam2_source`, `notes`.

### Связка камер ★
- **`observation`** — один куст в одном проходе, склейка cam1+cam2:
  `id`, `pass_id→pass`, `bush_id→bush`, `cam1_track_id`, `t_start`, `t_end`,
  `created_at`. Уникум: `(pass_id, bush_id, t_start)`.
- **`frame`** — кадр: `id`, `observation_id→observation`, `camera`
  (`cam1_bush`|`cam2_canopy`), `t`, `path`, `bbox` (json `[x,y,w,h]`).

### Результаты моделей (агрегаты на observation)
- **`vine_health`** — болезнь **лозы** (cam1): `disease_code`
  (`healthy`/`esca`/`eutypa`/`botryosphaeria`), `severity` (0..1), `confidence`.
- **`leaf_health`** — болезнь **листа** (cam2): `disease_code`
  (`healthy`/`mildew`/`oidium`/`black_rot`), `severity`, `confidence`.
- **`variety_prediction`** — сорт (cam2): `variety_id→variety`, `confidence`.
- **`inflorescence_stage`** — стадия соцветия (cam2): `stage_code` (BBCH), `confidence`.
- **`cluster_ripeness`** — зрелость грозди (cam2, сезонно): `ripeness_pct` (0..100),
  `brix_estimate`, `confidence`.
- **`cluster_health`** — болезнь грозди (cam2, сезонно): `disease_code`
  (`healthy`/`botrytis`/`sour_rot`/`berry_oidium`), `severity`, `confidence`.

Каждая из этих таблиц ссылается на `observation_id` и опционально на `frame_id`.

---

## Сквозной пример: куст V1-R03-B017, проход 3 июня 10:15

### Шаг 1 — справочники и топология (`main.py → run_pass`)

| Таблица | Строка |
|---|---|
| `vineyard` | `id=1, code='V1'` |
| `variety` | `id=1, code='cabernet_sauvignon'` |
| `vine_row` | `id=1, vineyard_id=1, variety_id=1, row_number=3` |
| `pass` | `id=1, row_id=1, started_at='2026-06-03T10:15', cam1_source='cam1.mp4', cam2_source='cam2.mp4'` |

### Шаг 2 — тележка едет, два потока идут параллельно
Камера 1 на 132.4 с фиксирует куст `track_id=57`. Пока трек не сменился, в него
стекаются: кадры камеры 1 (bbox + болезнь лозы) и кадры камеры 2
(сорт / лист / соцветие / гроздь).

### Шаг 3 — камера 1 переходит на следующий куст
Синхронизатор закрывает наблюдение, агрегирует и пишет:

| Таблица | Записанная строка | Источник |
|---|---|---|
| `bush` | `id=17, row_id=1, position_in_row=17, bush_code='V1-R03-B017'` | счётчик кустов |
| `observation` ★ | `id=4821, pass_id=1, bush_id=17, cam1_track_id=57, t_start=132.4, t_end=137.9` | синхронизатор |
| `frame` | `obs=4821, camera='cam1_bush', t=132.4, bbox=[10,20,50,80]` (×N) | cam1 |
| `frame` | `obs=4821, camera='cam2_canopy', t=132.5, ...` (×M) | cam2 |
| `vine_health` | `obs=4821, disease_code='esca', severity=0.30, confidence=0.85` | **cam1** |
| `variety_prediction` | `obs=4821, variety_id=1, confidence=0.97` | cam2 |
| `leaf_health` | `obs=4821, disease_code='mildew', severity=0.18, confidence=0.88` | cam2 |
| `inflorescence_stage` | `obs=4821, stage_code='BBCH-65', confidence=0.81` | cam2 |
| `cluster_ripeness` | `obs=4821, ripeness_pct=42.0, brix_estimate=14.5, confidence=0.76` | cam2 (сезонно) |
| `cluster_health` | `obs=4821, disease_code='botrytis', severity=0.12, confidence=0.79` | cam2 (сезонно) |

### Как читается потом — «всё про куст 17 в этот проход»

```sql
SELECT b.bush_code,
       vh.disease_code AS лоза,
       lh.disease_code AS лист,
       ch.disease_code AS гроздь,
       cr.ripeness_pct AS зрелость
FROM observation o
JOIN bush b                   ON b.id = o.bush_id
LEFT JOIN vine_health vh      ON vh.observation_id = o.id
LEFT JOIN leaf_health lh      ON lh.observation_id = o.id
LEFT JOIN cluster_health ch   ON ch.observation_id = o.id
LEFT JOIN cluster_ripeness cr ON cr.observation_id = o.id
WHERE b.bush_code = 'V1-R03-B017' AND o.pass_id = 1;
```

Результат: `V1-R03-B017 | esca | mildew | botrytis | 42.0`

---

## Ключевые принципы схемы

1. **`observation` — это «стык» двух камер.** Всё, что ниже неё, привязано к
   `observation_id`, а через него — к конкретному кусту и конкретному проходу.
   Это и есть зависимость «куст детектит cam1 → данные cam2 идут к нему».

2. **Сезонные таблицы пустуют вне сезона.** `cluster_ripeness` и `cluster_health`
   подключаются `LEFT JOIN`'ом: нет грозди — нет строки, запрос не ломается.

3. **История по времени бесплатно.** Тот же куст через 2 недели → новая
   `observation` с тем же `bush_id`, но другим `pass_id`. Динамика болезни и
   зрелости видна через `ORDER BY pass.started_at`.

4. **`severity` отделён от `disease_code`.** Класс болезни — *что*, severity (0..1) —
   *насколько*. «Здоров» хранится единообразно как `healthy`, severity=0.

## Три уровня болезней растения

| Сущность | Таблица | Камера | Когда |
|---|---|---|---|
| Болезнь **лозы** | `vine_health` | cam1 | всегда |
| Болезнь **листа** | `leaf_health` | cam2 | сезон вегетации |
| Болезнь **грозди** | `cluster_health` | cam2 | только когда есть гроздь |