import hashlib
import io
import json
import logging
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imagehash
import numpy as np
from PIL import Image

from .config import DEFAULT_NEAR_DUPLICATE_THRESHOLD


logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpeg", ".jpg", ".png"}


@dataclass
class SourceRecord:
    sample_id: str
    source_kind: str
    source_container: str
    source_name: str
    source_member: str
    width: int
    height: int
    phash: str
    is_duplicate: bool
    duplicate_reason: str
    duplicate_of: str
    pre_path: str = ""
    post_path: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _sample_id(container: str, member: str, source_name: str) -> str:
    token = f"{container}|{member}|{source_name}".encode("utf-8")
    return hashlib.sha1(token).hexdigest()[:16]


def _record_priority(source_kind: str) -> int:
    if source_kind == "filesystem":
        return 0
    if source_kind == "directory":
        return 1
    return 2


def _sort_key(p: Path) -> Tuple:
    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS:
        return (0, p.name.lower())
    elif p.is_dir() and not p.name.startswith("."):
        return (1, p.name.lower())
    elif p.is_file() and p.suffix.lower() == ".zip":
        return (2, p.name.lower())
    return (3, p.name.lower())


def iter_source_entries(source_dir: Path) -> Iterable[Tuple[str, Path, str, str]]:
    items = sorted(source_dir.iterdir(), key=_sort_key)
    for path in items:
        suffix = path.suffix.lower()
        if path.is_file() and suffix in VALID_EXTENSIONS:
            yield ("filesystem", path, path.name, "")
        elif path.is_file() and suffix == ".zip":
            with zipfile.ZipFile(path) as zf:
                members = sorted(
                    [name for name in zf.namelist() if Path(name).suffix.lower() in VALID_EXTENSIONS]
                )
                for member in members:
                    yield ("zip", path, Path(member).name, member)
        elif path.is_dir() and not path.name.startswith("."):
            dir_members = sorted(
                [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS],
                key=lambda f: f.name.lower(),
            )
            for member_path in dir_members:
                yield ("directory", path, member_path.name, member_path.name)


def load_image_bytes(source_kind: str, container: Path, member: str) -> bytes:
    if source_kind == "filesystem":
        return container.read_bytes()
    if source_kind == "directory":
        return (container / member).read_bytes()
    with zipfile.ZipFile(container) as zf:
        with zf.open(member) as fp:
            return fp.read()


def load_image(source_kind: str, container: Path, member: str) -> Image.Image:
    raw = load_image_bytes(source_kind, container, member)
    image = Image.open(io.BytesIO(raw))
    return image.convert("RGB")


def split_paired_image(image: Image.Image) -> Tuple[Image.Image, Image.Image]:
    width, height = image.size
    if width == height:
        midpoint = height // 2
        top = image.crop((0, 0, width, midpoint))
        bottom = image.crop((0, midpoint, width, height))
        return top, bottom
    midpoint = width // 2
    left = image.crop((0, 0, midpoint, height))
    right = image.crop((midpoint, 0, width, height))
    return left, right


def validate_split_halves(
    pre: Image.Image,
    post: Image.Image,
    sample_id: str,
    uniformity_threshold: float = 0.01,
) -> bool:
    valid = True
    for label, half in [("pre", pre), ("post", post)]:
        arr = np.asarray(half, dtype=np.float32)
        std = arr.std()
        if std < uniformity_threshold * 255:
            logger.warning(
                "Sample %s %s half appears nearly uniform (std=%.2f). Skipping.",
                sample_id, label, std,
            )
            valid = False
    return valid


def build_manifest(source_dir: Path, near_duplicate_threshold: int = DEFAULT_NEAR_DUPLICATE_THRESHOLD) -> List[SourceRecord]:
    entries = list(iter_source_entries(source_dir))
    entries.sort(key=lambda item: (_record_priority(item[0]), item[2].lower(), item[3].lower()))

    entry_data: List[Tuple[str, Path, str, str, Image.Image, imagehash.ImageHash, str]] = []
    for source_kind, container, source_name, member in entries:
        image = load_image(source_kind, container, member)
        phash_obj = imagehash.phash(image)
        phash_hex = str(phash_obj)
        entry_data.append((source_kind, container, source_name, member, image, phash_obj, phash_hex))

    n = len(entry_data)
    uf = _UnionFind()

    # Merge by duplicate name
    name_to_first: Dict[str, int] = {}
    for i, (_, _, source_name, _, _, _, _) in enumerate(entry_data):
        if source_name in name_to_first:
            uf.union(name_to_first[source_name], i)
        else:
            name_to_first[source_name] = i

    # Merge by phash similarity
    for i in range(n):
        for j in range(i + 1, n):
            distance = entry_data[i][5] - entry_data[j][5]
            if distance <= near_duplicate_threshold:
                uf.union(i, j)

    # Group into clusters
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    # Build records: pick one canonical per cluster
    records: List[SourceRecord] = []
    for _root, members in clusters.items():
        members.sort(key=lambda i: (_record_priority(entry_data[i][0]), entry_data[i][2].lower()))
        rep_idx = members[0]
        rep_sk, rep_cont, rep_sn, rep_mem, _, rep_phash, _ = entry_data[rep_idx]
        rep_id = _sample_id(str(rep_cont), rep_mem, rep_sn)

        for i in members:
            sk, cont, sn, mem, img, phash_obj, phash_hex = entry_data[i]
            sid = _sample_id(str(cont), mem, sn)

            if i == rep_idx:
                record = SourceRecord(
                    sample_id=sid, source_kind=sk, source_container=str(cont),
                    source_name=sn, source_member=mem,
                    width=img.width, height=img.height, phash=phash_hex,
                    is_duplicate=False, duplicate_reason="", duplicate_of="",
                )
            else:
                if sn in name_to_first and name_to_first[sn] != i and uf.find(name_to_first[sn]) == uf.find(i):
                    reason = "duplicate_name"
                else:
                    dist = phash_obj - rep_phash
                    reason = "duplicate_phash" if dist == 0 else f"near_duplicate_phash_{dist}"
                record = SourceRecord(
                    sample_id=sid, source_kind=sk, source_container=str(cont),
                    source_name=sn, source_member=mem,
                    width=img.width, height=img.height, phash=phash_hex,
                    is_duplicate=True, duplicate_reason=reason, duplicate_of=rep_id,
                )
            records.append(record)

    return records


def stable_split(sample_ids: List[str], seed: int, val_ratio: float, test_ratio: float) -> Dict[str, str]:
    ids = list(sample_ids)
    random.Random(seed).shuffle(ids)
    total = len(ids)
    test_count = max(1, int(total * test_ratio)) if total >= 10 else max(0, int(total * test_ratio))
    val_count = max(1, int(total * val_ratio)) if total >= 10 else max(0, int(total * val_ratio))
    train_count = max(0, total - test_count - val_count)

    split_map: Dict[str, str] = {}
    for sample_id in ids[:train_count]:
        split_map[sample_id] = "train"
    for sample_id in ids[train_count : train_count + val_count]:
        split_map[sample_id] = "val"
    for sample_id in ids[train_count + val_count :]:
        split_map[sample_id] = "test"
    return split_map


def write_summary(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
