"""Сборка YOLO-датасета для детектора ствола лозы (камера 1).

Источник (раскладка задачи, её наполняет synthetic.py или реальная разметка):
    dataset/annotations/vine_trunk_detection/
        images/<name>.jpg
        labels/<name>.txt        # YOLO: "<cls> cx cy w h" (cls=0 — ствол лозы)
        bush_index.csv           # name,bush_code  — связь кадра с кустом

Результат (то, что ест `yolo train data=...`):
    <out>/
        images/{train,val,test}/<name>.jpg
        labels/{train,val,test}/<name>.txt
        data.yaml

Ключевое: разрезаем по кусту/ряду через splits.split_samples, поэтому кадры
одного куста не растекаются по train и val (см. splits.py и DATASET.md).
"""
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .splits import Split, SplitRatios, _group_key, split_samples

TRUNK_TASK = "vine_trunk_detection"
INDEX_NAME = "bush_index.csv"
TRUNK_CLASSES = ["vine_trunk"]   # cls=0; болезнь лозы — отдельная задача/модель


# ----------------------------------------------------------------- раскладка источника
def trunk_src_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "annotations" / TRUNK_TASK


def trunk_label_dir(dataset_root: str | Path) -> Path:
    return trunk_src_dir(dataset_root) / "labels"


def append_index(dataset_root: str | Path, rows: list[tuple[str, str]]) -> None:
    """Дописать пары (name, bush_code) в bush_index.csv (создаёт заголовок)."""
    path = trunk_src_dir(dataset_root) / INDEX_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "bush_code"])
        w.writerows(rows)


# ----------------------------------------------------------------- чтение сэмплов
@dataclass(frozen=True)
class Sample:
    name: str
    image_path: Path
    label_path: Path
    bush_code: str


def read_samples(dataset_root: str | Path) -> list[Sample]:
    """Прочитать индекс и собрать сэмплы, у которых есть и картинка, и разметка."""
    src = trunk_src_dir(dataset_root)
    index = src / INDEX_NAME
    if not index.exists():
        raise FileNotFoundError(
            f"Нет {index}. Сначала наполните разметку (или python -m vinery.training.synthetic).")
    samples: list[Sample] = []
    with index.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name, code = row["name"], row["bush_code"]
            img = src / "images" / f"{name}.jpg"
            lbl = src / "labels" / f"{name}.txt"
            if img.exists() and lbl.exists():
                samples.append(Sample(name, img, lbl, code))
    return samples


# ----------------------------------------------------------------- сборка датасета
def _place(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if link:
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass  # нет прав на симлинк (типично для Windows) -> копируем
    shutil.copy2(src, dst)


def write_data_yaml(out_dir: Path, classes: list[str]) -> Path:
    """Записать data.yaml (вручную, без зависимости от pyyaml)."""
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(classes)],
        "",
    ]
    path = out_dir / "data.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_trunk_yolo_dataset(
    dataset_root: str | Path = "dataset",
    out_dir: str | Path = "dataset/yolo/vine_trunk",
    *,
    group_by: Literal["bush", "row"] = "row",
    ratios: SplitRatios = SplitRatios(),
    seed: int = 0,
    link: bool = False,
) -> dict:
    """Собрать YOLO-датасет из размеченных кадров. Вернуть сводку по сплитам.

    group_by='row' (по умолчанию) — строже всего: целый ряд целиком в одном сплите.
    link=True пытается делать симлинки (экономит место), иначе копирует файлы.
    """
    out_dir = Path(out_dir)
    samples = read_samples(dataset_root)
    if not samples:
        raise RuntimeError("Нет ни одного сэмпла с картинкой и разметкой.")

    buckets = split_samples(samples, key=lambda s: s.bush_code,
                            group_by=group_by, ratios=ratios, seed=seed)

    for split, items in buckets.items():
        for s in items:
            _place(s.image_path, out_dir / "images" / split / f"{s.name}.jpg", link)
            _place(s.label_path, out_dir / "labels" / split / f"{s.name}.txt", link)

    yaml_path = write_data_yaml(out_dir, TRUNK_CLASSES)
    _assert_no_leakage(buckets, group_by)

    summary = {
        "out_dir": str(out_dir.resolve()),
        "data_yaml": str(yaml_path.resolve()),
        "group_by": group_by,
        "counts": {sp: len(items) for sp, items in buckets.items()},
        "groups": {sp: len({_group_key(s.bush_code, group_by) for s in items})
                   for sp, items in buckets.items()},
    }
    return summary


def _assert_no_leakage(buckets: dict[Split, list[Sample]],
                       group_by: Literal["bush", "row"]) -> None:
    """Страховка: ни одна группа (куст/ряд) не должна попасть в два сплита."""
    seen: dict[str, str] = {}
    for split, items in buckets.items():
        for s in items:
            gk = _group_key(s.bush_code, group_by)
            if gk in seen and seen[gk] != split:
                raise AssertionError(
                    f"Утечка: группа {gk} в '{seen[gk]}' и '{split}' одновременно.")
            seen[gk] = split


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка YOLO-датасета детекции лозы (cam1).")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--out", default="dataset/yolo/vine_trunk")
    ap.add_argument("--group-by", choices=["bush", "row"], default="row")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--link", action="store_true", help="симлинки вместо копий")
    args = ap.parse_args()
    summary = build_trunk_yolo_dataset(
        args.root, args.out, group_by=args.group_by,
        ratios=SplitRatios(args.train, args.val, args.test),
        seed=args.seed, link=args.link)
    print("Готов YOLO-датасет:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()