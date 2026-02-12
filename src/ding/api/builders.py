"""Factories for constructing denoisers and samplers from config dictionaries."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from omegaconf import DictConfig, OmegaConf

from ding.api.base import BaseSampler
from ding.denoisers import AVAILABLE_DENOISERS, BaseDenoiser
from ding.samplers import (
    DingSampler,
    FlairSampler,
    FlowChefSampler,
    DiffPIRSampler,
    DDNMSampler,
    BlendedDiffusionSampler,
)


def _to_dict(
    config: Mapping[str, Any] | DictConfig | None
) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)

def build_denoiser(
    config: Mapping[str, Any] | DictConfig,
    device: str,
    dtype: torch.dtype | None = None,
) -> BaseDenoiser:
    cache_dir = config.get("cache_dir") if isinstance(config, Mapping) else getattr(config, "cache_dir", None)
    print("Using cache dir:", cache_dir)
    
    name = config.get("name") if isinstance(config, Mapping) else config.name
    if name not in AVAILABLE_DENOISERS:
        raise ValueError(f"Unknown denoiser '{name}'. Available: {sorted(AVAILABLE_DENOISERS)}")

    params = _to_dict(config.get("params") if isinstance(config, Mapping) else config.params)

    params.setdefault("device", device)
    params.setdefault("cache_dir", cache_dir)
    if dtype is not None and "dtype" not in params:
        params["dtype"] = dtype

    denoiser_cls = AVAILABLE_DENOISERS[name]
    return denoiser_cls(**params)


def build_sampler(
    config: Mapping[str, Any] | DictConfig,
    denoiser: BaseDenoiser,
    device: str,
) -> BaseSampler:
    name = config.get("name") if isinstance(config, Mapping) else config.name

    if name == "ding":
        return DingSampler(denoiser, device=device)
    if name == "flair":
        return FlairSampler(denoiser, device=device)
    if name == "flow_chef":
        return FlowChefSampler(denoiser, device=device)
    if name == "diffpir":
        return DiffPIRSampler(denoiser, device=device)
    if name == "ddnm":
        return DDNMSampler(denoiser, device=device)
    if name == "blended_diffusion":
        return BlendedDiffusionSampler(denoiser, device=device)

    raise ValueError(
        f"Unknown sampler '{name}'. Available: "
        "['flow_edit', 'ding', 'flair', 'flow_chef', 'diffpir', 'ddnm', 'blended_diffusion']"
    )
