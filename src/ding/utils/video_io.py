"""Video IO helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch import Tensor
from PIL import Image

from diffusers.utils import load_video
from diffusers.utils.export_utils import export_to_video

_FRAME_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".gif", ".webm", ".mkv"}


def _natural_key(value: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", value.stem)
    key: list[int | str] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        elif part:
            key.append(part.lower())
    key.append(value.suffix.lower())
    return key

def list_frame_paths(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Video frames directory not found: {directory}")
    frame_paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _FRAME_EXTENSIONS
    ]
    if not frame_paths:
        raise ValueError(f"No frame images found in directory: {directory}")
    frame_paths.sort(key=_natural_key)
    return frame_paths

def _pil_to_tensor(
    image: Image.Image,
    device: str,
    dtype: torch.dtype,
    target_size: Sequence[int] | None,
) -> Tensor:
    if target_size is not None:
        image = image.resize((target_size[1], target_size[0]), resample=Image.BICUBIC)

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device=device, dtype=dtype)

def _load_video_file(
    path: Path,
    device: str,
    dtype: torch.dtype,
    target_size: Sequence[int] | None,
    target_num_frames: int | None,
) -> Tensor:

    pil_frames = load_video(str(path))
    if len(pil_frames) == 0:
        raise ValueError(f"No frames decoded from video file: {path}")

    target_frames = pil_frames[:target_num_frames]
    frame_tensors = [
        _pil_to_tensor(frame, device=device, dtype=dtype, target_size=target_size) for frame in target_frames
    ]
    return torch.stack(frame_tensors, dim=1)

def load_and_resize_video(
    directory: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    target_height: int | None = None,
    target_width: int | None = None,
    target_num_frames: int | None = None,
) -> Tensor:
    """Load a directory of RGB frames into a tensor of shape ``(C, F, H, W)`` in ``[-1, 1]``."""

    dir_path = Path(directory)
    if dir_path.is_file() and dir_path.suffix.lower() in VIDEO_EXTENSIONS:
        target_size: Sequence[int] | None = None
        if target_height is not None and target_width is not None:
            target_size = (int(target_height), int(target_width))
        elif target_height is not None or target_width is not None:
            raise ValueError("Both target_height and target_width must be provided when resizing video files.")
        return _load_video_file(
            dir_path,
            device=device,
            dtype=dtype,
            target_size=target_size,
            target_num_frames=target_num_frames,
        )

    frame_paths = list_frame_paths(dir_path)
    selected = frame_paths[:target_num_frames]

    target_size: Sequence[int] | None = None
    if target_height is not None or target_width is not None:
        height = target_height if target_height is not None else None
        width = target_width if target_width is not None else None
        if height is None or width is None:
            raise ValueError("Both target_height and target_width must be provided.")
        target_size = (int(height), int(width))

    frames: list[Tensor] = []
    for path in selected:
        frame = _load_video_file(
            path,
            device=device,
            dtype=dtype,
            target_size=target_size,
        )
        frames.append(frame)

    stacked = torch.stack(frames, dim=1)  # (C, F, H, W)
    return stacked

def save_tensor_as_video(
    video: Tensor,
    path: str | Path,
    frame_rate: int = 24,
) -> Path:
    """Persist a video tensor to disk as an MP4."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("save_tensor_as_video only supports a single-sample batch.")
        video = video.squeeze(0)

    if video.ndim != 4:
        raise ValueError("Video tensor must have shape (C, F, H, W) or (1, C, F, H, W).")

    frames = video.permute(1, 2, 3, 0).detach().to(dtype=torch.float32, device="cpu")
    if frames.shape[-1] == 1:
        frames = frames.repeat(1, 1, 1, 3)
    frames = (frames.clamp(-1.0, 1.0) + 1.0) * 0.5  # to [0,1]
    frames_np = frames.numpy()
    pil_frames = [Image.fromarray((frame * 255.0).astype(np.uint8)) for frame in frames_np]
    export_to_video(pil_frames, output_video_path=str(path), fps=frame_rate)
    return path

def persist_video_outputs(
    out_dir: Path,
    observation: Tensor,
    reconstructions: Tensor,
    frame_rate: int,
) -> Dict[str, Any]:
    obs_path = out_dir / "observation.mp4"
    save_tensor_as_video(observation, obs_path, frame_rate=frame_rate)

    recon_paths: list[Path] = []
    for idx, sample in enumerate(reconstructions):
        recon_path = out_dir / ("recon.mp4" if idx == 0 else f"recon_{idx}.mp4")
        save_tensor_as_video(sample, recon_path, frame_rate=frame_rate)
        recon_paths.append(recon_path)

    files: Dict[str, Any] = {
        "observation": obs_path,
        "reconstruction": recon_paths[0],
    }
    if len(recon_paths) > 1:
        files["reconstruction_samples"] = tuple(recon_paths)

    return files