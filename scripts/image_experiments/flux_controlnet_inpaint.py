"""Run FLUX ControlNet inpainting over a dataset directory.

Expected dataset layout:
  data_dir/
    image00005.png
    mask00005.png
    metadata.json

metadata.json entries (example):
  {
    "image": "image00005.png",
    "mask": "mask00005.png",
    "inpaint_prompt": "A pizza sitting on a table with flowers"
  }

Mask convention default: 1 = context, 0 = inpaint region.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from diffusers.utils import load_image, check_min_version
check_min_version("0.30.2") # Need diffusers v0.30.2 for FluxControlNetInpaintingPipeline

from alimama_flux_controlnet_inpaint.controlnet_flux import FluxControlNetModel
from alimama_flux_controlnet_inpaint.transformer_flux import FluxTransformer2DModel
from alimama_flux_controlnet_inpaint.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline


def _load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in metadata.json, got {type(data).__name__}.")
    return [entry for entry in data if isinstance(entry, dict)]

def _resolve_entries(data_dir: Path, metadata_path: Path) -> list[dict[str, Any]]:
    entries = []
    for entry in _load_metadata(metadata_path):
        image_name = entry.get("image")
        mask_name = entry.get("mask")
        if not image_name or not mask_name:
            continue
        entries.append(
            {
                "image_path": data_dir / str(image_name),
                "mask_path": data_dir / str(mask_name),
                "prompt": str(entry.get("inpaint_prompt", "") or ""),
            }
        )
    return entries

def _prepare_sizes(
    image: Image.Image, mask: Image.Image, width: int | None, height: int | None
) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    if width is None or height is None:
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        return image, mask, image.size

    target_size = (width, height)
    if image.size != target_size:
        image = image.resize(target_size, Image.BICUBIC)
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.NEAREST)
    return image, mask, target_size


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUX ControlNet inpainting on a dataset.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Dataset root containing images, masks, metadata.json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.1-dev",
        help="FLUX model id or local path.",
    )
    parser.add_argument(
        "--controlnet",
        default="alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
        help="ControlNet model id or local path.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device to run on (e.g., cuda, cuda:0, cpu).")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
        help="Torch dtype for the model.",
    )
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--true-guidance-scale", type=float, default=1.0, help="default: 3.5 for alpha and 1.0 for beta")
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=0.90)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--width", type=int, default=1024, help="Target width (optional).")
    parser.add_argument("--height", type=int, default=1024, help="Target height (optional).")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (offset by index).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of items to process.")
    args = parser.parse_args()
    if (args.width is None) ^ (args.height is None):
        parser.error("Pass both --width and --height, or neither.")
    return args


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir
    metadata_path = data_dir / "metadata.json"
    entries = _resolve_entries(data_dir, metadata_path)
    if not entries:
        raise RuntimeError(f"No entries found in {metadata_path} or {data_dir}.")

    dtype = getattr(torch, args.dtype)
    controlnet = FluxControlNetModel.from_pretrained(
        args.controlnet,
        use_safetensors=True,
        extra_conditioning_channels=1,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model,
        subfolder='transformer',
        torch_dtype=dtype,
    )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        args.model,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.transformer.to(dtype)
    pipe.controlnet.to(dtype)
    pipe.to(args.device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(entries)} entries from {metadata_path}...")
    for index, entry in enumerate(entries):
        if args.limit is not None and index >= args.limit:
            break

        image_path = Path(entry["image_path"])
        mask_path = Path(entry["mask_path"])
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue
        if not mask_path.exists():
            print(f"Skipping missing mask: {mask_path}")
            continue

        out_name = f"{image_path.stem}{image_path.suffix}"
        out_path = output_dir / out_name

        prompt = entry.get("prompt")
        image = load_image(str(image_path))
        orig_mask = load_image(str(mask_path)) # convention 0 = inpaint region
        mask = Image.fromarray(255 - np.array(orig_mask))
        image, mask, target_size = _prepare_sizes(image, mask, args.width, args.height)

        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=args.device).manual_seed(args.seed + index)

        with torch.inference_mode():
            result = pipe(
                negative_prompt="",
                prompt=prompt,
                height=target_size[1],
                width=target_size[0],
                control_image=image,
                control_mask=mask,
                num_inference_steps=args.steps,
                generator=generator,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                guidance_scale=args.guidance_scale,
                true_guidance_scale=args.true_guidance_scale
            ).images[0]
        result.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
