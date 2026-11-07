"""Atomic checkpoints with pruning (last K plus milestones) and a free-disk guard."""
from __future__ import annotations

import os
import pathlib
import re
import shutil

import torch

MIN_FREE_GB = float(os.environ.get("NUC_MIN_FREE_GB", "50"))  # shared NUC floor (NUC_RULES.md)


def _name(it: int) -> str:
    return f"ckpt_{it:08d}.pt"


def save_atomic(state: dict, dirpath: str | pathlib.Path, iteration: int) -> pathlib.Path:
    d = pathlib.Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(d).free / 1e9
    if free_gb < MIN_FREE_GB:
        raise RuntimeError(f"refusing to write a checkpoint: only {free_gb:.0f} GB free (< {MIN_FREE_GB})")
    final = d / _name(iteration)
    tmp = d / (final.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    dfd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return final


def list_ckpts(dirpath) -> list[tuple[int, pathlib.Path]]:
    d = pathlib.Path(dirpath)
    out = []
    if d.is_dir():
        for p in d.iterdir():
            m = re.fullmatch(r"ckpt_(\d{8})\.pt", p.name)
            if m:
                out.append((int(m.group(1)), p))
    return sorted(out)


def load_latest(dirpath) -> dict | None:
    """Newest checkpoint that loads; a corrupt/partial newest file falls back to the previous one."""
    for it, p in reversed(list_ckpts(dirpath)):
        try:
            return torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"[ckpt] skipping unreadable {p.name}: {e}", flush=True)
    return None


def prune(dirpath, keep_last: int = 3, milestone_every: int = 500, max_milestones: int = 20):
    cks = list_ckpts(dirpath)
    last = {it for it, _ in cks[-keep_last:]}
    miles = [it for it, _ in cks if it % milestone_every == 0 and it not in last]
    keep_m = set(miles[-max_milestones:])
    for it, p in cks:
        if it not in last and it not in keep_m:
            p.unlink(missing_ok=True)
    for p in pathlib.Path(dirpath).glob("*.tmp*"):  # leftovers from a killed save
        if p.stat().st_mtime < os.path.getmtime(dirpath) - 3600:
            p.unlink(missing_ok=True)


def dir_size_mb(dirpath) -> float:
    return sum(p.stat().st_size for p in pathlib.Path(dirpath).glob("*") if p.is_file()) / 1e6
