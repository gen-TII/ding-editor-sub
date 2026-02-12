"""Audio IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torchaudio
from torch import Tensor
    

def load_and_resize_audio(
    path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    target_length: int | None = None,
    target_sample_rate: int | None = None,
) -> Tensor:
    path = Path(path)

    audio, original_sample_rate = torchaudio.load(path)  # (C, T), float32 CPU

    # Enforce stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] != 2:
        raise ValueError(f"Audio at {path} has {audio.shape[0]} channels, expected 1 or 2.")

    # Optional resampling
    if target_sample_rate is not None and original_sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(
            audio, original_sample_rate, target_sample_rate
        )

    # Resize in time (CPU)
    if target_length is not None:
        current_length = audio.shape[-1]
        if current_length < target_length:
            pad_amount = target_length - current_length
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
        elif current_length > target_length:
            audio = audio[..., :target_length]

    # Move to target device / dtype last
    audio = audio.to(device=device, dtype=dtype)

    return audio

def save_tensor_as_audio(
    tensor: Tensor,
    path: str | Path,
    sample_rate: int = 44100,
) -> None:
    """Persist a tensor in [-1, 1] to disk as audio.

    Accepted shapes:
      - (T,)         mono
      - (C, T)       channels, samples
      - (B, C, T)    batch, channels, samples
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tensor = tensor.detach()

    # Normalize shapes
    if tensor.ndim == 1:
        # (T,) -> (1, 1, T)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 2:
        # (C, T) -> (1, C, T)
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        # (B, C, T), ok
        pass
    else:
        raise ValueError(f"Expected 1D, 2D or 3D tensor, got shape {tuple(tensor.shape)}")

    tensor_cpu = tensor.to(dtype=torch.float32, device="cpu")
    scaled = tensor_cpu.clamp(-1.0, 1.0)

    for idx in range(scaled.shape[0]):
        out_path = (
            str(path)
            if scaled.shape[0] == 1
            else str(path.with_name(f"{path.stem}_{idx}{path.suffix}"))
        )
        torchaudio.save(
            uri=out_path,
            src=scaled[idx],  # (C, T) float32 CPU clamped
            sample_rate=sample_rate,
            channels_first=True,
        )

def persist_audio_outputs(
    out_dir: Path,
    observation: Tensor,
    reconstructions: Tensor,
    sample_rate: int = 44100,
) -> Dict[str, Any]:
    obs_path = out_dir / "observation.wav"
    recon_path = out_dir / "recon.wav"
    save_tensor_as_audio(observation, obs_path, sample_rate)
    save_tensor_as_audio(reconstructions, recon_path, sample_rate)

    recon_paths = [
        recon_path if idx == 0 else out_dir / f"recon_{idx}.wav"
        for idx in range(reconstructions.shape[0])
    ]

    files: Dict[str, Any] = {
        "observation": obs_path,
        "reconstruction": recon_paths[0],
    }
    if len(recon_paths) > 1:
        files["reconstruction_samples"] = tuple(recon_paths)

    return files
