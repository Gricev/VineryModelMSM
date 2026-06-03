"""split-by-bush: главное свойство — НЕТ утечки кадров одного куста между сплитами,
и распределение детерминированно (стабильно между запусками)."""
from dataclasses import dataclass

from vinery.training import assign_split, row_of, split_samples
from vinery.training.splits import SplitRatios


@dataclass(frozen=True)
class S:
    bush_code: str


def _samples():
    # 6 рядов по 20 кустов, по 5 кадров на куст
    out = []
    for row in range(1, 7):
        for slot in range(1, 21):
            code = f"V1-R{row:02d}-B{slot:03d}"
            out += [S(code) for _ in range(5)]
    return out


def test_row_of():
    assert row_of("V1-R03-B017") == "V1-R03"


def test_no_leakage_group_by_row():
    buckets = split_samples(_samples(), key=lambda s: s.bush_code, group_by="row")
    seen = {}
    for split, items in buckets.items():
        for s in items:
            rk = row_of(s.bush_code)
            assert seen.setdefault(rk, split) == split  # ряд в одном сплите


def test_no_leakage_group_by_bush():
    buckets = split_samples(_samples(), key=lambda s: s.bush_code, group_by="bush")
    seen = {}
    for split, items in buckets.items():
        for s in items:
            assert seen.setdefault(s.bush_code, split) == split
    # 120 кустов -> при таком объёме групп баланс надёжен, все сплиты непусты
    assert all(buckets[s] for s in ("train", "val", "test"))


def test_deterministic():
    a = assign_split("V1-R03", SplitRatios())
    b = assign_split("V1-R03", SplitRatios())
    assert a == b


def test_ratios_must_sum_to_one():
    import pytest
    with pytest.raises(ValueError):
        SplitRatios(0.5, 0.4, 0.4)