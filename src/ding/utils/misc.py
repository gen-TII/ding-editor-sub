"""General-purpose utilities (seeding, environment inspection, etc.)."""

from __future__ import annotations

from pathlib import Path

import random
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

import numpy as np
import torch

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

def set_deterministic_seed(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

def abs_path(value) -> Path | None:
    if value in (None, ""):
        return None
    value_path = Path(str(value)).expanduser()
    if not value_path.is_absolute():
        value_path = Path(get_original_cwd()) / value_path
    return value_path.resolve()

def banner(message: str) -> str:
    line = "=" * len(message)
    return f"{line}\n{message}\n{line}"


def resolve_bbox(bbox_cfg) -> tuple[int, int, int, int] | None:
    if bbox_cfg in (None, ""):
        return None
    bbox = list(bbox_cfg)
    if len(bbox) != 4:
        raise ValueError("Bounding box must contain four integers (x0, y0, x1, y1).")
    return tuple(int(v) for v in bbox)

def resolve_audio_bbox(bbox_cfg) -> tuple[float, float] | None:
    if bbox_cfg in (None, ""):
        return None
    bbox = list(bbox_cfg)
    if len(bbox) != 2:
        raise ValueError("Bounding box must contain two floats (start_in_s, end_in_s).")
    return tuple(int(v) for v in bbox)

def resolve_dtype(name: str | torch.dtype | None) -> torch.dtype | None:
    if name is None:
        return None
    if isinstance(name, torch.dtype):
        return name
    try:
        return getattr(torch, name)
    except AttributeError as exc:  # pragma: no cover
        raise ValueError(f"Unknown torch dtype '{name}'.") from exc


def resolve_prompts(conditioning: DictConfig | None) -> list[str]:
    if conditioning is None:
        return [""]

    raw = conditioning.get("prompts", "")
    if isinstance(raw, str):
        prompts = [raw]
    else:
        prompts = [str(p) for p in raw]

    if len(prompts) == 0:
        prompts = [""]
    return prompts