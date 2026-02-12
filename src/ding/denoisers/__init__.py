"""Denoiser wrappers and factory helpers."""

from .base import BaseDenoiser, SchedulerTensors
from . import sd3 as _sd3
from .flux import FluxDenoiser
from .ltx import LTXVideoDenoiser
from .wan import Wan21Denoiser
from .sa1 import StableAudio1Denoiser

StableDiffusion3Denoiser = _sd3.StableDiffusion3Denoiser
SD3MediumDenoiser = _sd3.SD3MediumDenoiser
SD35MediumDenoiser = _sd3.SD35MediumDenoiser
SD35LargeDenoiser = _sd3.SD35LargeDenoiser
SD35LargeTurboDenoiser = _sd3.SD35LargeTurboDenoiser


AVAILABLE_DENOISERS = {
    "sd3_medium": SD3MediumDenoiser,
    "sd3.5_medium": SD35MediumDenoiser,
    "sd3.5_large": SD35LargeDenoiser,
    "sd3.5_large_turbo": SD35LargeTurboDenoiser,
    "flux": FluxDenoiser,
    "ltx": LTXVideoDenoiser,
    "wan": Wan21Denoiser,
    "sa1": StableAudio1Denoiser,
}

__all__ = [
    "BaseDenoiser",
    "SchedulerTensors",
    "StableDiffusion3Denoiser",
    "SD3MediumDenoiser",
    "SD35MediumDenoiser",
    "SD35LargeDenoiser",
    "SD35LargeTurboDenoiser",
    "FluxDenoiser",
    "LTXVideoDenoiser",
    "Wan21Denoiser",
    "StableAudio1Denoiser",
    "AVAILABLE_DENOISERS",
]
