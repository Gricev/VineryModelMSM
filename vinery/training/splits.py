"""Разрезание датасета на train/val/test БЕЗ утечки между кустами.

Почему это отдельный модуль и почему так важно (см. DATASET.md §«split по кусту»):
кадры одного куста почти идентичны (тот же ствол, та же лоза, соседние кадры
одного проезда). Если они попадут и в train, и в val — модель «увидит ответ»
на валидации, метрики взлетят, а в поле модель провалится. Поэтому сплит режется
не по кадру, а по ГРУППЕ: все кадры одного куста (а лучше — целого ряда) обязаны
оказаться в одном и том же сплите.

Распределение детерминированное (стабильный хеш ключа группы), поэтому:
  - один и тот же куст всегда попадает в один и тот же сплит между запусками;
  - добавление новых кустов не перетасовывает уже распределённые;
  - не нужно хранить файлы splits/*.txt, чтобы воспроизвести разбиение.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitRatios:
    """Доли train/val/test. Должны давать в сумме 1.0."""
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Доли сплита должны давать 1.0, а дают {total}")


def row_of(bush_code: str) -> str:
    """Ключ ряда из кода куста: 'V1-R03-B017' -> 'V1-R03'.

    Группировка по ряду строже, чем по кусту: соседние кусты ряда снимаются
    в одном проезде в близких условиях (свет, фон, сорт — один на ряд), поэтому
    утечка «через соседа» тоже возможна. По умолчанию рекомендуется group_by='row'.
    """
    parts = bush_code.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else bush_code


def _group_key(bush_code: str, group_by: Literal["bush", "row"]) -> str:
    return row_of(bush_code) if group_by == "row" else bush_code


def assign_split(group_key: str, ratios: SplitRatios = SplitRatios(),
                 seed: int = 0) -> Split:
    """Детерминированно отнести группу к train/val/test.

    Берём стабильный хеш (sha1, не встроенный hash() — он рандомизирован между
    процессами через PYTHONHASHSEED) и проецируем в [0, 1). Граница по долям.
    """
    digest = hashlib.sha1(f"{seed}:{group_key}".encode("utf-8")).hexdigest()
    x = int(digest[:8], 16) / 0xFFFFFFFF
    if x < ratios.train:
        return "train"
    if x < ratios.train + ratios.val:
        return "val"
    return "test"


def split_samples(
    samples: Iterable,
    *,
    key: Callable[[object], str],
    group_by: Literal["bush", "row"] = "row",
    ratios: SplitRatios = SplitRatios(),
    seed: int = 0,
) -> dict[Split, list]:
    """Разложить произвольные сэмплы по сплитам, группируя по кусту/ряду.

    key      — функция, достающая bush_code из сэмпла;
    group_by — 'row' (по умолчанию, строже) или 'bush'.
    Гарантия: все сэмплы с одинаковым ключом группы попадают в один сплит.
    """
    out: dict[Split, list] = {"train": [], "val": [], "test": []}
    for s in samples:
        gk = _group_key(key(s), group_by)
        out[assign_split(gk, ratios, seed)].append(s)
    return out