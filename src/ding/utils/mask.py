"""Utility functions for handling masks in image, video, and audio inpainting tasks."""

from __future__ import annotations

from pathlib import Path
from PIL import Image
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def _validate_bbox(
    bbox: Iterable[int],
    width: int,
    height: int
) -> Tuple[int, int, int, int]:
    """Validate bounding box coordinates."""
    coords = list(map(int, bbox))
    if len(coords) != 4:
        raise ValueError("Bounding box must have four integer coordinates (x0, y0, x1, y1).")

    x0, y0, x1, y1 = coords
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            "Bounding box coordinates must satisfy 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height."
        )
    return x0, y0, x1, y1

def _mask_from_bbox(
    height: int,
    width: int,
    bbox: Iterable[int],
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    invert: bool = False,
) -> Tensor:
    """Create a binary mask (1=observed, 0=unknown) from a bounding box."""

    x0, y0, x1, y1 = _validate_bbox(bbox, width=width, height=height)

    mask = torch.ones(1, height, width, device=device, dtype=dtype)
    mask[..., y0:y1, x0:x1] = 0.0

    if invert:
        mask = 1 - mask

    return mask

def _load_image_mask(
    path: str | Path,
    image_shape: Optional[tuple[int, int]] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    invert: bool = False,
) -> Tensor:
    """Load a image mask."""

    path = Path(path)
    if path.suffix in {".pt", ".pth"}:
        mask = torch.load(path, map_location=device)
        mask = torch.as_tensor(mask, device=device, dtype=dtype)
    elif path.suffix == ".npy":
        mask = torch.from_numpy(np.load(path)).to(device=device, dtype=dtype)
    else:
        image = Image.open(path).convert("L")
        if image_shape is not None:
            image = image.resize((image_shape[1], image_shape[0]), Image.NEAREST)
        mask_array = np.asarray(image, dtype=np.float32) / 255.0
        mask = torch.from_numpy(mask_array).to(device=device, dtype=dtype).unsqueeze(0)

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3 and mask.shape[0] != 1:
        mask = mask.mean(dim=0, keepdim=True)

    if invert:
        mask = 1 - mask

    if image_shape is not None and mask.shape[-2:] != image_shape:
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0),
            size=image_shape,
            mode="nearest",
        ).squeeze(0)

    return mask.clamp(0.0, 1.0).to(device=device, dtype=dtype)

def resolve_image_mask(
    mask_path: Optional[str | Path],
    bbox: Optional[tuple[int, int, int, int]],
    image_shape: tuple[int, int],
    device: str,
    dtype: torch.dtype,
) -> Tensor:
    if mask_path is not None:
        return _load_image_mask(mask_path, image_shape=image_shape, device=device, dtype=dtype)

    if bbox is not None:
        h, w = image_shape
        return _mask_from_bbox(
            height=h,
            width=w,
            bbox=bbox,
            device=device,
            dtype=dtype,
        )

    print("\n...No mask or bbox provided – using full ones mask.")
    h, w = image_shape
    return torch.ones(1, h, w, device=device, dtype=dtype)

def load_video_mask(
    path: str | Path | None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    target_height: int | None = None,
    target_width: int | None = None,
    target_num_frames: int | None = None,
    invert: bool = False,
) -> Tensor:
    """Load a video mask."""

    if path in (None, ""):
        print("Mask path is None, we procede with all ones mask")
        return torch.ones(1, target_num_frames, target_height, target_width, device=device, dtype=dtype)

    mask_path = Path(path)

    if mask_path.suffix == ".npy":
        mask_tensor = torch.from_numpy(np.load(mask_path)).to(device=device, dtype=dtype)
        mask_tensor = mask_tensor[:target_num_frames]
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor.unsqueeze(0)

    if mask_tensor.shape[-2:] != (target_height, target_width):
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0),
            size=(target_num_frames, target_height, target_width),
            mode="trilinear",
        ).squeeze(0)

    if invert:
        mask_tensor = 1 - mask_tensor

    return mask_tensor.clamp(0.0, 1.0).to(device=device, dtype=dtype)

def _load_audio_mask(
    path: str | Path | None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    target_length: int | None = None,
    invert: bool = False,
) -> Tensor:
    """Load audio mask."""

    mask_path = Path(path)

    if mask_path.suffix == ".npy":
        mask_tensor = torch.from_numpy(np.load(mask_path)).to(device=device, dtype=dtype)
    else:
        raise ValueError("Unsupported mask file format. Only .npy files are supported for audio masks.")

    if target_length is not None:
        current_length = mask_tensor.shape[-1]
        if current_length < target_length:
            pad_amount = target_length - current_length
            mask_tensor = torch.nn.functional.pad(mask_tensor, (1, pad_amount))
        elif current_length > target_length:
            mask_tensor = mask_tensor[..., :target_length]

    if mask_tensor.ndim == 1:
        mask_tensor = mask_tensor.unsqueeze(0)
    elif mask_tensor.ndim != 2:
        raise ValueError("Audio mask must be a 1D or 2D tensor.")

    if invert:
        mask_tensor = 1 - mask_tensor

    return mask_tensor.clamp(0.0, 1.0).to(device=device, dtype=dtype)

def _audio_mask_from_bbox(
    bbox,
    target_length: int,
    sample_rate: int,
    device="cpu",
    dtype=torch.float32
) -> Tensor:
    start_s, end_s = map(float, bbox)
    start_idx = int(round(start_s * sample_rate))
    end_idx   = int(round(end_s   * sample_rate))
    start_idx = max(0, min(start_idx, target_length))
    end_idx   = max(0, min(end_idx,   target_length))
    if end_idx <= start_idx:
        raise ValueError("Invalid (start_in_s, end_in_s) interval.")
    mask = torch.ones(1, target_length, device=device, dtype=dtype)
    mask[..., start_idx:end_idx] = 0.0
    return mask


def resolve_audio_mask(
    mask_path: Optional[str | Path],
    bbox: Optional[tuple[float, float]],
    target_length: int | None = None,
    sample_rate: int = 44100,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if mask_path is not None:
        return _load_audio_mask(mask_path, target_length=target_length, device=device, dtype=dtype)

    if bbox is not None:
        print("Mask path is None, we procede with bbox mask")
        return _audio_mask_from_bbox(
            bbox=bbox,
            target_length=target_length,
            sample_rate=sample_rate,
            device=device,
            dtype=dtype,
        )

    print("Mask path is None, bbox is None, we procede with all ones mask")
    return torch.ones(1, target_length, device=device, dtype=dtype)
    

def resize_mask_for_latent(
    mask: torch.Tensor,
    latent_shape: tuple,
    threshold: float = 0.95,
    dilate: bool = False,
    kernel_spatial: int = 3,
    kernel_time: int = 1,
) -> Tensor:
    """
    Resize image/video mask to match latent spatial (and temporal) size.
    - Audio latent shape : (B, C, L)
    - Image latent shape : (B, C, H, W)
    - Video latent shape : (B, C, F, H, W)
    """
    if len(latent_shape) == 3:
        # Audio masks are defined over time while the latent temporal
        # dimension is downsampled by a fixed factor via hop-aligned pooling.
        # We binarize then take min over each of the L contiguous time blocks.
        target_length = int(latent_shape[-1])

        if mask.ndim == 2:
            mask = mask.unsqueeze(1)  # (B, 1, T)
        elif mask.ndim == 3:
            if mask.shape[1] != 1:
                mask = mask.mean(dim=1, keepdim=True)
        else:
            raise ValueError("Audio mask must be a 1D, 2D, or 3D tensor.")

        mask = (mask > threshold).float()
        time_length = int(mask.shape[-1])
        if time_length != target_length:
            hop = (time_length + target_length - 1) // target_length
            padded_length = hop * target_length
            pad_right = padded_length - time_length
            if pad_right:
                mask = torch.nn.functional.pad(mask, (0, pad_right), value=1.0)
            mask = mask.reshape(mask.shape[0], 1, target_length, hop).amin(dim=-1)

    elif len(latent_shape) == 4:
        # Image masks are defined over pixel space while the latent spatial
        # dimensions are downsampled by a fixed factor. We use bilinear interpolation
        # with antialiasing to resize the mask.
        mask = torch.nn.functional.interpolate(
            mask,
            size=latent_shape[-2:],
            mode="bilinear",
            antialias=True
        )

    elif len(latent_shape) == 5:
        # Video masks are defined over pixel space while the latent spatial
        # dimensions are downsampled by a fixed factor. We use trilinear interpolation
        # to resize the mask.
        mask = torch.nn.functional.interpolate(
            mask,
            size=latent_shape[-3:],
            mode="trilinear"
        )
        
    else:
        raise ValueError("latent_shape must be (B,C,L) for audio or (B,C,H,W) for image or (B,C,F,H,W) video")

    mask = (mask > threshold).float()

    # dilate mask for overestimation
    if dilate:
        inv = 1 - mask
        if len(latent_shape) == 3:
            # audio : inv [B, C, L]
            pad = kernel_spatial // 2
            inv = torch.nn.functional.max_pool1d(inv, kernel_size=kernel_spatial, stride=1, padding=pad)
        elif len(latent_shape) == 4:
            # image : inv [B, C, H, W]
            pad = kernel_spatial // 2
            inv = torch.nn.functional.max_pool2d(inv, kernel_size=kernel_spatial, stride=1, padding=pad)
        else:
            # vidéo : inv [B, C, F, H, W]
            pad_t = kernel_time // 2
            pad_s = kernel_spatial // 2
            inv = torch.nn.functional.max_pool3d(
                inv,
                kernel_size=(kernel_time, kernel_spatial, kernel_spatial),
                stride=1,
                padding=(pad_t, pad_s, pad_s),
            )
        mask = 1 - inv

    return mask
