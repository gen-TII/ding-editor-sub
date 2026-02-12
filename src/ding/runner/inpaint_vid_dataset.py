"""Hydra-ready runner for video dataset inpainting."""

from __future__ import annotations

import torch 
import gc
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from ding.api.builders import build_denoiser, build_sampler
from ding.utils import (
    CONFIG_DIR,
    set_deterministic_seed,
    abs_path,
    banner,
    resolve_dtype,
    save_tensor_as_video,
) 

def _configure_denoiser(
    denoiser,
    cfg: DictConfig,
    prompts: list[str],
    n_videos_per_prompt: int,
    steps: int,
    target_height: int,
    target_width: int,
    target_num_frames: int,
    frame_rate: float,
) -> None:

    # Set video params
    denoiser.set_video_params(
        batch_size=n_videos_per_prompt*len(prompts),
        height=target_height,
        width=target_width,
        num_frames=target_num_frames,
        frame_rate=frame_rate,
    )

    # Set prompts
    conditioning = cfg.get("conditioning")
    guidance_scale = 3.0
    guidance_rescale = 0.0
    negative_prompts = None

    if conditioning is not None:
        guidance_scale = float(conditioning.get("guidance_scale", guidance_scale))
        guidance_rescale = float(conditioning.get("guidance_rescale", guidance_rescale))
        negative_prompts = conditioning.get("negative_prompts", negative_prompts)

    denoiser.set_prompt(
        prompts=prompts,
        negative_prompts=negative_prompts,
        guidance_scale=guidance_scale,
        guidance_rescale=guidance_rescale,
        n_videos_per_prompt=n_videos_per_prompt,
    )

    # Set timesteps
    denoiser.set_timesteps(steps)

@hydra.main(config_path=str(CONFIG_DIR), config_name="inpaint_vid_dataset", version_base=None)
def main(cfg: DictConfig) -> None:

    print(banner("ding.runner.inpaint_vid_dataset"))
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # set params from config
    show_progress = bool(cfg.get("show_progress", True))
    seed = int(cfg.get("seed", 0))
    set_deterministic_seed(seed)
    device = cfg.device
    dtype = resolve_dtype(cfg.get("model_dtype"))
    steps = int(cfg.get("steps", 28))
    dilate_mask = bool(cfg.get("dilate_mask", True))
    conditioning_cfg = cfg.get("conditioning")
    n_videos_per_prompt = conditioning_cfg.get("n_videos_per_prompt", 1) if conditioning_cfg else 1
    n_videos_per_prompt = max(1, int(n_videos_per_prompt))
    target_height = cfg.video.get("target_height")
    target_width = cfg.video.get("target_width")
    target_num_frames = cfg.video.get("target_num_frames")
    frame_rate = cfg.video.get("frame_rate")

    # build denoiser and sampler
    denoiser = build_denoiser(cfg.denoiser, device=device, dtype=dtype)
    sampler = build_sampler(cfg.sampler, denoiser=denoiser, device=device)

    # resolve paths
    dataset_dir = abs_path(cfg.dataset_dir)
    if dataset_dir is None:
        raise ValueError("dataset_dir must be provided")
    metadata_file = abs_path(cfg.metadata_file)
    if metadata_file is None:
        raise ValueError("metadata_file must be provided")
    output_dir = abs_path(cfg.output_dir)
    if output_dir is None:
        raise ValueError("output_dir must be provided")
    output_dir = output_dir / f"{cfg.sampler.name}_{cfg.denoiser.name}_{steps}s_{conditioning_cfg.guidance_scale}gs"

    # load metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    for item in metadata:
        video_path = dataset_dir / item["video"]
        assert video_path.exists(), video_path
        mask_path = dataset_dir / item["mask"]
        assert mask_path.exists(), mask_path
        inpaint_prompt = item["inpaint_prompt"]
        prompts_list = [inpaint_prompt] if isinstance(inpaint_prompt, str) else inpaint_prompt
        n_samples = len(prompts_list) * n_videos_per_prompt

        _configure_denoiser(
            denoiser,
            cfg=cfg,
            prompts=prompts_list,
            n_videos_per_prompt=n_videos_per_prompt,
            steps=steps,
            target_height=target_height,
            target_width=target_width,
            target_num_frames=target_num_frames,
            frame_rate=frame_rate,
        )

        # sample videos
        artifacts = sampler.sample_video(
            video_path=video_path,
            mask_path=mask_path,
            out_dir=output_dir,
            steps=steps,
            n_samples=n_samples,
            target_height=target_height,
            target_width=target_width,
            target_num_frames=target_num_frames,
            frame_rate=frame_rate,
            show_progress=show_progress,
            seed=seed,
            save_outputs=False,
            dilate_mask=dilate_mask,
        )
        artifacts.to_cpu()
        gen_video_path = output_dir / video_path.name
        save_tensor_as_video(
            artifacts.reconstructions,
            gen_video_path,
        )
        print(banner("Inpainted video saved to " + str(gen_video_path)))

        del artifacts
        torch.cuda.empty_cache()
        gc.collect()

    print(banner("Run complete"))


if __name__ == "__main__":
    main()

