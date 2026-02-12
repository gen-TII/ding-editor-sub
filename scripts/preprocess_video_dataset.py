"""Preprocess a video dataset with resized videos, masks, and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ding.utils import load_and_resize_video, load_video_mask, save_tensor_as_video


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess video dataset")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-height", type=int, required=True)
    parser.add_argument("--target-width", type=int, required=True)
    parser.add_argument("--target-num-frames", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--output-metadata", type=str, default="metadata.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    input_dir = args.input_dir
    metadata_path = args.metadata
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(metadata_path.read_text())
    new_metadata = []

    for item in metadata:
        print(f"Processing video ID: {item.get('video_id', 'N/A')}")
        video_path = _resolve_path(input_dir, item.get("video_path"))
        mask_path = _resolve_path(input_dir, item.get("mask_path"))
        prompt = item.get("video_inpainting_prompt")
        video_id = item.get("video_id")

        video = load_and_resize_video(
            video_path,
            device="cpu",
            dtype=torch.float32,
            target_height=args.target_height,
            target_width=args.target_width,
            target_num_frames=args.target_num_frames,
        )

        num_frames = int(video.shape[1])
        height = int(video.shape[2])
        width = int(video.shape[3])

        mask = load_video_mask(
            mask_path,
            device="cpu",
            dtype=torch.float32,
            target_height=height,
            target_width=width,
            target_num_frames=num_frames,
        )

        video_name = f"{video_id}.mp4"
        mask_name = f"{video_id}.npy"

        video_out = output_dir / video_name
        mask_out = output_dir / mask_name

        save_tensor_as_video(video, video_out, frame_rate=args.fps)
        np.save(mask_out, mask.squeeze(0).cpu().numpy())

        new_metadata.append(
            {
                "video": str(video_out.relative_to(output_dir)),
                "mask": str(mask_out.relative_to(output_dir)),
                "inpaint_prompt": prompt,
            }
        )

    out_metadata_path = output_dir / args.output_metadata
    out_metadata_path.write_text(json.dumps(new_metadata, indent=2))
    print(f"Saved metadata to {out_metadata_path}")


if __name__ == "__main__":
    main()
