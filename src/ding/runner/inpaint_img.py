"""Hydra-ready runner for image inpainting."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from ding.api.builders import build_denoiser, build_sampler
from ding.utils import (
    CONFIG_DIR,
    set_deterministic_seed,
    abs_path,
    banner,
    resolve_bbox,
    resolve_dtype,
    resolve_prompts,
) 


def _configure_denoiser(
    denoiser,
    cfg: DictConfig,
    prompts: list[str],
    n_images_per_prompt: int,
    steps: int,
    target_height: int,
    target_width: int,
) -> None:
    
    # Set image params
    denoiser.set_image_params(
        height=target_height,
        width=target_width,
    )

    # Set prompts
    conditioning = cfg.get("conditioning")
    guidance_scale = 3.0
    negative_prompts = None
    true_cfg_scale = 1.0
    
    if conditioning is not None:
        guidance_scale = float(conditioning.get("guidance_scale", guidance_scale))
        negative_prompts = conditioning.get("negative_prompts")
        true_cfg_scale = float(conditioning.get("true_cfg_scale", true_cfg_scale))
    
    denoiser.set_prompt(
        prompts=prompts,
        negative_prompts=negative_prompts,
        guidance_scale=guidance_scale,
        n_images_per_prompt=n_images_per_prompt,
        true_cfg_scale=true_cfg_scale,
    )
    
    # Set timesteps
    denoiser.set_timesteps(steps)


@hydra.main(config_path=str(CONFIG_DIR), config_name="inpaint_img", version_base=None)
def main(cfg: DictConfig) -> None:
    
    print(banner("ding.runner.inpaint_img"))
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # set params from config
    show_progress = bool(cfg.get("show_progress", True))
    seed = int(cfg.get("seed", 0))
    set_deterministic_seed(seed)
    device = cfg.device
    dtype = resolve_dtype(cfg.get("model_dtype"))
    steps = int(cfg.get("steps", 28))
    conditioning_cfg = cfg.get("conditioning")
    prompts = resolve_prompts(conditioning_cfg)
    bbox = resolve_bbox(cfg.get("bbox"))
    n_images_per_prompt = conditioning_cfg.get("n_images_per_prompt", 1) if conditioning_cfg else 1
    n_images_per_prompt = max(1, int(n_images_per_prompt))
    n_samples = max(1, len(prompts) * n_images_per_prompt)
    target_height = int(cfg.image.get("target_height", 1024))
    target_width = int(cfg.image.get("target_width", 1024))

    # build denoiser and sampler
    denoiser = build_denoiser(cfg.denoiser, device=device, dtype=dtype)
    _configure_denoiser(
        denoiser,
        cfg=cfg,
        prompts=prompts,
        n_images_per_prompt=n_images_per_prompt,
        steps=steps,
        target_height=target_height,
        target_width=target_width,
    )
    sampler = build_sampler(cfg.sampler, denoiser=denoiser, device=device)

    # resolve paths
    image_path = abs_path(cfg.image_path)
    if image_path is None:
        raise ValueError("image_path must be provided")
    mask_path = abs_path(cfg.mask_path)
    output_dir = abs_path(cfg.output_dir)
    if output_dir is None:
        raise ValueError("output_dir must be provided")
    output_dir.mkdir(parents=True, exist_ok=True)

    # sample images
    artifacts = sampler.sample_image(
        image_path=image_path,
        mask_path=mask_path,
        out_dir=output_dir,
        steps=steps,
        n_samples=n_samples,
        target_height=target_height,
        target_width=target_width,
        bbox=bbox,
        show_progress=show_progress,
        seed=seed,
        save_outputs=True,
    )
    artifacts.to_cpu()
    print(banner("Run complete"))
    print(f"Artifacts stored in: {artifacts.output_dir}")


if __name__ == "__main__":
    main()
