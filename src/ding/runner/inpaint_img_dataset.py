"""Hydra-ready runner for image dataset inpainting."""

from __future__ import annotations

import json
import hydra
from omegaconf import DictConfig, OmegaConf

from ding.api.builders import build_denoiser, build_sampler
from ding.utils import (
    CONFIG_DIR,
    set_deterministic_seed,
    abs_path,
    banner,
    resolve_dtype,
    save_tensor_as_image
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


@hydra.main(config_path=str(CONFIG_DIR), config_name="inpaint_img_dataset", version_base=None)
def main(cfg: DictConfig) -> None:
    
    print(banner("ding.runner.inpaint_img_dataset"))
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # set params from config
    show_progress = bool(cfg.get("show_progress", True))
    seed = int(cfg.get("seed", 0))
    set_deterministic_seed(seed)
    device = cfg.device
    dtype = resolve_dtype(cfg.get("model_dtype"))
    steps = int(cfg.get("steps", 28))
    conditioning_cfg = cfg.get("conditioning")
    n_images_per_prompt = conditioning_cfg.get("n_images_per_prompt", 1) if conditioning_cfg else 1
    n_images_per_prompt = max(1, int(n_images_per_prompt))
    target_height = int(cfg.image.get("target_height", 1024))
    target_width = int(cfg.image.get("target_width", 1024))

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
        image_id = item["image"]
        image_path = dataset_dir / image_id
        assert image_path.exists(), image_path
        mask = item["mask"]
        mask_path = dataset_dir / mask
        assert mask_path.exists(), mask_path
        inpaint_prompt = item["inpaint_prompt"]
        prompts_list = [inpaint_prompt] if isinstance(inpaint_prompt, str) else inpaint_prompt
        n_samples = len(prompts_list) * n_images_per_prompt

        _configure_denoiser(
            denoiser,
            cfg=cfg,
            prompts=inpaint_prompt,
            n_images_per_prompt=n_images_per_prompt,
            steps=steps,
            target_height=target_height,
            target_width=target_width,
        )
        
        # sample images
        artifacts = sampler.sample_image(
            image_path=image_path,
            mask_path=mask_path,
            out_dir=output_dir,
            steps=steps,
            n_samples=n_samples,
            target_height=target_height,
            target_width=target_width,
            show_progress=show_progress,
            seed=seed,
            save_outputs=False,
        )
        artifacts.to_cpu()
        save_tensor_as_image(
            artifacts.reconstructions,
            output_dir / image_id
        )
        print(banner("Inpainted image saved to " + str(output_dir / image_id)))

    print(banner("Run complete"))

if __name__ == "__main__":
    main()
