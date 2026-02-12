"""Sampler abstractions and artefact containers."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass(slots=True)
class RunArtifacts:
    """Container returned by samplers.

    Attributes
    ----------
    reconstructions:
        Tensor of reconstructed samples in ``[-1, 1]``.
    output_dir:
        Directory where artefacts were persisted.
    metadata:
        Arbitrary JSON-serialisable metadata describing the run.
    files:
        Mapping of logical artefact names to on-disk paths.
    """

    reconstructions: torch.Tensor
    output_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, Any] = field(default_factory=dict)

    def to_cpu(self) -> RunArtifacts:
        """Return a shallow copy with reconstructions moved to CPU."""

        return RunArtifacts(
            reconstructions=self.reconstructions.detach().cpu(),
            output_dir=self.output_dir,
            metadata=dict(self.metadata),
            files=dict(self.files),
        )


class BaseSampler(abc.ABC):
    """Interface that all samplers must implement."""

    def __init__(self, *, device: Optional[str] = None) -> None:
        self._device = device or "cpu"

    @property
    def device(self) -> str:
        return self._device

    @abc.abstractmethod
    def sample_image(
        self,
        image_path: str | Path,
        mask_path: Optional[str | Path],
        out_dir: str | Path,
        steps: int,
        **kwargs: Any,
    ) -> RunArtifacts:
        """Run image posterior sampling and persist artefacts."""

    @abc.abstractmethod
    def sample_video(
        self,
        video_path: str | Path,
        mask_path: Optional[str | Path],
        out_dir: str | Path,
        steps: int,
        **kwargs: Any,
    ) -> RunArtifacts:
        """Run video posterior sampling and persist artefacts."""

    @abc.abstractmethod
    def sample_audio(
        self,
        audio_path: str | Path,
        mask_path: Optional[str | Path],
        out_dir: str | Path,
        steps: int,
        **kwargs: Any,
    ) -> RunArtifacts:
        """Run audio posterior sampling and persist artefacts."""

    def _prepare_output_dir(self, out_dir: str | Path) -> Path:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
