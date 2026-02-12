"""Hydra-ready runner for video inpainting."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from ding.api.builders import build_denoiser, build_sampler
from ding.utils import (
    CONFIG_DIR,
    set_deterministic_seed,
    abs_path,
    banner,
    resolve_dtype,
    resolve_prompts,
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

@hydra.main(config_path=str(CONFIG_DIR), config_name="inpaint_vid", version_base=None)
def main(cfg: DictConfig) -> None:

    print(banner("ding.runner.inpaint_vid"))
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
    prompts = resolve_prompts(conditioning_cfg)
    n_videos_per_prompt = conditioning_cfg.get("n_videos_per_prompt", 1) if conditioning_cfg else 1
    n_videos_per_prompt = max(1, int(n_videos_per_prompt))
    n_samples = max(1, len(prompts) * n_videos_per_prompt)
    target_height = cfg.video.get("target_height")
    target_width = cfg.video.get("target_width")
    target_num_frames = cfg.video.get("target_num_frames")
    frame_rate = cfg.video.get("frame_rate")

    # build denoiser and sampler
    denoiser = build_denoiser(cfg.denoiser, device=device, dtype=dtype)
    _configure_denoiser(
        denoiser,
        cfg=cfg,
        prompts=prompts,
        n_videos_per_prompt=n_videos_per_prompt,
        steps=steps,
        target_height=target_height,
        target_width=target_width,
        target_num_frames=target_num_frames,
        frame_rate=frame_rate,
    )
    sampler = build_sampler(cfg.sampler, denoiser=denoiser, device=device)

    # resolve paths
    video_path = abs_path(cfg.video_path)
    if video_path is None:
        raise ValueError("video_path must be provided")
    mask_path = abs_path(cfg.mask_path)
    output_dir = abs_path(cfg.output_dir)
    if output_dir is None:
        raise ValueError("output_dir must be provided")
    output_dir.mkdir(parents=True, exist_ok=True)

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
        save_outputs=True,
        dilate_mask=dilate_mask,
    )
    artifacts.to_cpu()
    print(banner("Run complete"))
    print(f"Artifacts stored in: {artifacts.output_dir}")


if __name__ == "__main__":
    main()

