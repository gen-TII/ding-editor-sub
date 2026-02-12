"""Image IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def load_and_resize_image(
    path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    target_height: int = None,
    target_width: int = None,
) -> Tensor:
    """Load an RGB image, resize it, and normalise it to ``[-1, 1]``."""

    image = Image.open(path).convert("RGB")
    image = image.resize((target_width, target_height), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device=device, dtype=dtype)


def save_tensor_as_image(
    tensor: Tensor,
    path: str | Path
) -> None:
    """Persist a tensor in ``[-1, 1]`` to disk as an image."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tensor = tensor.detach()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)

    tensor_cpu = tensor.to(dtype=torch.float32, device="cpu")
    min_val = tensor_cpu.min().item()
    max_val = tensor_cpu.max().item()

    if min_val >= 0.0 and max_val <= 1.0:
        scaled = tensor_cpu.clamp(0.0, 1.0) * 255.0
    else:
        scaled = (tensor_cpu.clamp(-1.0, 1.0) + 1.0) * 127.5

    array = scaled.to(torch.uint8).permute(0, 2, 3, 1).numpy()

    def _save(idx: int, img: np.ndarray, target: Path) -> None:
        if img.shape[-1] == 1:
            Image.fromarray(img[..., 0], mode="L").save(target)
        else:
            Image.fromarray(img, mode="RGB").save(target)

    if array.shape[0] == 1:
        _save(0, array[0], path)
    else:
        for idx, img in enumerate(array):
            target = path if idx == 0 else path.with_name(f"{path.stem}_{idx}{path.suffix}")
            _save(idx, img, target)


def persist_image_outputs(
    out_dir: Path,
    observation: Tensor,
    reconstructions: Tensor,
) -> Dict[str, Any]:
    obs_path = out_dir / "observation.png"
    recon_path = out_dir / "recon.png"
    save_tensor_as_image(observation, obs_path)
    save_tensor_as_image(reconstructions, recon_path)

    recon_paths = [
        recon_path if idx == 0 else out_dir / f"recon_{idx}.png"
        for idx in range(reconstructions.shape[0])
    ]

    files: Dict[str, Any] = {
        "observation": obs_path,
        "reconstruction": recon_paths[0],
    }
    if len(recon_paths) > 1:
        files["reconstruction_samples"] = tuple(recon_paths)

    return files