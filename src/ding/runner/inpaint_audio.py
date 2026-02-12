"""Hydra-ready runner for audio inpainting."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from ding.api.builders import build_denoiser, build_sampler
from ding.utils import (
    CONFIG_DIR,
    set_deterministic_seed,
    abs_path,
    banner,
    resolve_audio_bbox, 
    resolve_dtype,
    resolve_prompts,
) 


def _configure_denoiser(
    denoiser,
    cfg: DictConfig,
    prompts: list[str],
    n_wavs_per_prompt: int,
    steps: int,
    audio_start_in_s: float,
    audio_end_in_s: float,
) -> None:
    
    # Set prompts
    conditioning = cfg.get("conditioning")
    guidance_scale = 7.0
    negative_prompts = None
    
    if conditioning is not None:
        guidance_scale = float(conditioning.get("guidance_scale", 7.0))
        negative_prompts = conditioning.get("negative_prompts")
    
    denoiser.set_prompt(
        prompts=prompts,
        negative_prompts=negative_prompts,
        guidance_scale=guidance_scale,
    )
    
    # Set audio params (run after setting prompts)
    denoiser.set_audio_params(
        audio_start_in_s=audio_start_in_s,
        audio_end_in_s = audio_end_in_s,
        n_wavs_per_prompt=n_wavs_per_prompt,
    )

    # Set timesteps
    denoiser.set_timesteps(steps)


@hydra.main(config_path=str(CONFIG_DIR), config_name="inpaint_audio", version_base=None)
def main(cfg: DictConfig) -> None:
    
    print(banner("ding.runner.inpaint_audio"))
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
    bbox = resolve_audio_bbox(cfg.get("bbox"))
    n_wavs_per_prompt = conditioning_cfg.get("n_wavs_per_prompt", 1) if conditioning_cfg else 1
    n_wavs_per_prompt = max(1, int(n_wavs_per_prompt))
    n_samples = max(1, len(prompts) * n_wavs_per_prompt)
    audio_start_in_s = int(cfg.audio.get("audio_start_in_s", 0.0))
    audio_end_in_s = int(cfg.audio.get("audio_end_in_s", 10.0))

    # build denoiser and sampler
    denoiser = build_denoiser(cfg.denoiser, device=device, dtype=dtype)
    _configure_denoiser(
        denoiser,
        cfg=cfg,
        prompts=prompts,
        n_wavs_per_prompt=n_wavs_per_prompt,
        steps=steps,
        audio_start_in_s=audio_start_in_s,
        audio_end_in_s=audio_end_in_s,
    )
    sampler = build_sampler(cfg.sampler, denoiser=denoiser, device=device)

    # resolve paths
    audio_path = abs_path(cfg.audio_path)
    if audio_path is None:
        raise ValueError("audio_path must be provided")
    mask_path = abs_path(cfg.mask_path)
    output_dir = abs_path(cfg.output_dir)
    if output_dir is None:
        raise ValueError("output_dir must be provided")
    output_dir.mkdir(parents=True, exist_ok=True)

    # sample audios
    artifacts = sampler.sample_audio(
        audio_path=audio_path,
        mask_path=mask_path,
        out_dir=output_dir,
        steps=steps,
        n_samples=n_samples,
        audio_start_in_s=audio_start_in_s,
        audio_end_in_s=audio_end_in_s,
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
