#!/usr/bin/env python3
"""Run Wan VACE inpainting over a video dataset.

Expected dataset layout:
  data_dir/
    vid_id.mp4
    mask_id.npy
    metadata.json

metadata.json entries (example):
  {
    "video": "vid_id.mp4",
    "mask": "mask_id.npy",
    "inpaint_prompt": "..."
  }

Mask convention default: 1 = context (keep), 0 = inpaint region.
Use --mask-mode inpaint if your mask already marks inpaint regions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_video


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)


def _load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in metadata.json, got {type(data).__name__}.")
    return [entry for entry in data if isinstance(entry, dict)]


def _resolve_entries(data_dir: Path, metadata_path: Path) -> list[dict[str, Any]]:
    entries = []
    for entry in _load_metadata(metadata_path):
        video_name = entry.get("video")
        mask_name = entry.get("mask")
        if not video_name or not mask_name:
            continue
        entries.append(
            {
                "video_path": data_dir / str(video_name),
                "mask_path": data_dir / str(mask_name),
                "prompt": str(entry.get("inpaint_prompt", "") or ""),
            }
        )
    return entries


def _load_video_frames(
    video_path: Path,
    target_size: tuple[int, int] | None,
    target_num_frames: int | None,
) -> list[Image.Image]:
    frames = load_video(str(video_path))
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}.")
    if target_num_frames is not None:
        frames = frames[:target_num_frames]
    if target_size is not None:
        frames = [frame.resize(target_size, Image.BICUBIC) for frame in frames]
    return frames


def _load_mask_frames(
    mask_path: Path,
    target_size: tuple[int, int],
    target_num_frames: int | None,
    mask_mode: str,
    mask_threshold: float | None,
    dilate: bool,
    kernel_time: int,
    kernel_spatial: int,
) -> list[Image.Image]:
    mask = np.load(mask_path).astype(np.float32)
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim != 3:
        raise ValueError(f"Unsupported mask shape {mask.shape} in {mask_path}.")
    if mask.max() > 1.0:
        mask = mask / 255.0
    if target_num_frames is not None:
        mask = mask[:target_num_frames]

    if mask_mode == "context":
        mask = 1.0 - mask

    if dilate:
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        pad_t = kernel_time // 2
        pad_s = kernel_spatial // 2
        mask_tensor = torch.nn.functional.max_pool3d(
            mask_tensor,
            kernel_size=(kernel_time, kernel_spatial, kernel_spatial),
            stride=1,
            padding=(pad_t, pad_s, pad_s),
        )
        mask = mask_tensor.squeeze(0).squeeze(0).numpy()

    if mask_threshold is not None:
        mask = (mask >= mask_threshold).astype(np.float32)
    mask = np.clip(mask, 0.0, 1.0)

    frames: list[Image.Image] = []
    for frame in mask:
        image = Image.fromarray((frame * 255.0).round().astype(np.uint8), mode="L")
        if image.size != target_size:
            image = image.resize(target_size, Image.NEAREST)
        frames.append(image)
    return frames


def _prepare_video_and_mask(
    video_path: Path,
    mask_path: Path,
    target_size: tuple[int, int] | None,
    target_num_frames: int | None,
    mask_mode: str,
    mask_threshold: float | None,
    dilate: bool,
    kernel_time: int,
    kernel_spatial: int,
) -> tuple[list[Image.Image], list[Image.Image], tuple[int, int], int]:
    video_frames = _load_video_frames(video_path, target_size, target_num_frames)
    if not video_frames:
        raise ValueError(f"No video frames available for {video_path}.")

    resolved_size = target_size or video_frames[0].size
    mask_frames = _load_mask_frames(
        mask_path,
        target_size=resolved_size,
        target_num_frames=target_num_frames,
        mask_mode=mask_mode,
        mask_threshold=mask_threshold,
        dilate=dilate,
        kernel_time=kernel_time,
        kernel_spatial=kernel_spatial,
    )

    if len(video_frames) != len(mask_frames):
        print(
            f"Aligning frame counts for {video_path.name}: "
            f"video={len(video_frames)}, mask={len(mask_frames)}."
        )
    num_frames = min(len(video_frames), len(mask_frames))
    video_frames = video_frames[:num_frames]
    mask_frames = mask_frames[:num_frames]

    if num_frames == 0:
        raise ValueError(f"No overlapping frames for {video_path}.")
    return video_frames, mask_frames, resolved_size, num_frames


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wan VACE inpainting on a dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Dataset root containing videos, masks, and metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save outputs (defaults to <data-dir>/wan_vace_outputs).",
    )
    parser.add_argument(
        "--model",
        default="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        help="Wan VACE model id or local path.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device to run on (e.g., cuda, cuda:0, cpu).")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
        help="Torch dtype for the model.",
    )
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=928, help="Target width (optional).")
    parser.add_argument("--height", type=int, default=512, help="Target height (optional).")
    parser.add_argument("--num-frames", type=int, default=97, help="Target number of frames (optional).")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the output video.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed (offset by index). Use -1 for nondeterministic sampling.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to process.")
    parser.add_argument(
        "--mask-mode",
        choices=("context", "inpaint"),
        default="context",
        help="Interpretation of mask values: context=1 means keep, inpaint=1 means fill.",
    )
    parser.add_argument(
        "--dilate-mask",
        type=bool,
        default=True,
        help="Dilate the inpaint region mask (after mask-mode conversion).",
    )
    parser.add_argument(
        "--dilate-kernel-time",
        type=int,
        default=1,
        help="Temporal kernel size for mask dilation.",
    )
    parser.add_argument(
        "--dilate-kernel-spatial",
        type=int,
        default=3,
        help="Spatial kernel size for mask dilation.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Optional threshold in [0,1] to binarize mask values.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt applied to every sample.",
    )
    args = parser.parse_args()
    if (args.width is None) ^ (args.height is None):
        parser.error("Pass both --width and --height, or neither.")
    if args.num_frames is not None and args.num_frames <= 0:
        parser.error("--num-frames must be positive when provided.")
    if args.mask_threshold is not None and not (0.0 <= args.mask_threshold <= 1.0):
        parser.error("--mask-threshold must be in [0,1].")
    if args.dilate_mask:
        if args.dilate_kernel_time < 1 or args.dilate_kernel_spatial < 1:
            parser.error("--dilate-kernel-time and --dilate-kernel-spatial must be >= 1.")
    return args


def main() -> None:
    args = _parse_args()
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    data_dir = args.data_dir
    metadata_path = data_dir / "metadata.json"
    entries = _resolve_entries(data_dir, metadata_path)
    if not entries:
        raise RuntimeError(f"No entries found in {metadata_path} or {data_dir}.")

    dtype = getattr(torch, args.dtype)
    vae = AutoencoderKLWan.from_pretrained(
        args.model,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanVACEPipeline.from_pretrained(args.model, vae=vae, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
    pipe.to(args.device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_size = None
    if args.width is not None and args.height is not None:
        target_size = (args.width, args.height)

    print(f"Processing {len(entries)} entries from {metadata_path}...")
    for index, entry in enumerate(entries):
        if args.limit is not None and index >= args.limit:
            break

        video_path = Path(entry["video_path"])
        mask_path = Path(entry["mask_path"])
        if not video_path.exists():
            print(f"Skipping missing video: {video_path}")
            continue
        if not mask_path.exists():
            print(f"Skipping missing mask: {mask_path}")
            continue

        video_frames, mask_frames, resolved_size, num_frames = _prepare_video_and_mask(
            video_path,
            mask_path,
            target_size=target_size,
            target_num_frames=args.num_frames,
            mask_mode=args.mask_mode,
            mask_threshold=args.mask_threshold,
            dilate=args.dilate_mask,
            kernel_time=args.dilate_kernel_time,
            kernel_spatial=args.dilate_kernel_spatial,
        )

        fill = Image.new("RGB", resolved_size, (0, 0, 0))
        video_frames = [
            Image.composite(fill, frame, mask.convert("L"))
            for frame, mask in zip(video_frames, mask_frames)
        ]

        prompt = entry.get("prompt", "")

        with torch.inference_mode():
            result = pipe(
                video=video_frames,
                mask=mask_frames,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=resolved_size[1],
                width=resolved_size[0],
                num_frames=num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).frames[0]

        out_path = output_dir / f"{video_path.stem}.mp4"
        export_to_video(result, str(out_path), fps=args.fps)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
