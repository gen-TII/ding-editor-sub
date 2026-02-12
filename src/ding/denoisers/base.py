"""Common base classes for denoisers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass(slots=True)
class SchedulerTensors:
    alphas: Tensor
    sigmas: Tensor
    alphas_f64: Tensor
    sigmas_f64: Tensor


class BaseDenoiser(torch.nn.Module):
    """Minimal interface required by samplers."""

    def __init__(
        self,
        alphas_f64: Tensor,
        sigmas_f64: Tensor,
        timesteps: Tensor,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()

        self.alphas_f64 = alphas_f64
        self.sigmas_f64 = sigmas_f64
        self.timesteps = timesteps
        self.dtype = dtype
        self.device = device

        self.alphas = alphas_f64.to(device, dtype)
        self.sigmas = sigmas_f64.to(device, dtype)

    # ------------------------------------------------------------------
    # API expected by Samplers
    # ------------------------------------------------------------------

    def pred_velocity(self, x: Tensor, t: Tensor | int) -> Tensor:
        raise NotImplementedError

    def pred_x0(self, x: Tensor, t: Tensor | int) -> Tensor:
        v_pred = self.pred_velocity(x, t)
        return x - self.sigmas[t] * v_pred

    def pred_x1(self, x: Tensor, t: Tensor | int) -> Tensor:
        v_pred = self.pred_velocity(x, t)
        return x + self.alphas[t] * v_pred

    def encode(self, x: Tensor) -> Tensor:
        return x

    def decode(self, x: Tensor) -> Tensor:
        return x
