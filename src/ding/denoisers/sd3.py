"""Stable Diffusion 3 family image denoiser wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import Tensor
import numpy as np

from diffusers.utils.torch_utils import randn_tensor
from .base import BaseDenoiser
from diffusers import StableDiffusion3Pipeline


class StableDiffusion3Denoiser(BaseDenoiser):
    """Wrapper around the SD3 image generation pipeline."""

    default_model_id: str = "stabilityai/stable-diffusion-3-medium-diffusers"

    def __init__(
        self,
        device: str = "cuda:0",
        dtype: torch.dtype | str = torch.float16,
        model_id: str | Path | None = None,
        cache_dir: str | Path | None = None,
        model_cpu_offload: bool = False,
        sequential_cpu_offload: bool = False,
        attention_slicing: bool = False,
        vae_slicing: bool = False,
    ) -> None:
        
        if isinstance(dtype, str):
            if not hasattr(torch, dtype):
                raise ValueError(f"Unknown torch dtype '{dtype}'.")
            dtype = getattr(torch, dtype)

        model_id = str(model_id or self.default_model_id)
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "cache_dir": cache_dir}
        pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, **load_kwargs)

        pipeline, device_obj = self._setup_pipeline_optimizations(
            pipeline=pipeline,
            device=device,
            enable_model_cpu_offload=model_cpu_offload,
            enable_sequential_cpu_offload=sequential_cpu_offload,
            enable_attention_slicing=attention_slicing,
            enable_vae_slicing=vae_slicing,
        )

        text_encoder = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2
        text_encoder_3 = pipeline.text_encoder_3
        transformer = pipeline.transformer
        vae = pipeline.vae
        scheduler = pipeline.scheduler

        scale_factor = pipeline.vae_scale_factor
        vae_scale_factor = vae.config.scaling_factor
        vae_shift_factor = vae.config.shift_factor
        height = width = pipeline.default_sample_size * scale_factor
        n_channels = transformer.config.in_channels

        timesteps_desc = scheduler.timesteps.to(device=device_obj, dtype=torch.float32)
        sigmas_desc = scheduler.sigmas.to(device=device_obj, dtype=torch.float64)
        net_timesteps = timesteps_desc.flip(0)
        sigmas_f64 = sigmas_desc.flip(0)
        alphas_f64 = (1.0 - sigmas_f64).clamp(min=0.0)
        num_train_timesteps = scheduler.config.num_train_timesteps
        timesteps = torch.arange(num_train_timesteps, dtype=torch.int64, device=device_obj)

        super().__init__(
            alphas_f64=alphas_f64,
            sigmas_f64=sigmas_f64,
            timesteps=timesteps,
            dtype=dtype,
            device=device_obj,
        )

        self.pipeline = pipeline
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.scheduler = scheduler
        self.transformer = transformer
        self.vae = vae
    
        self.device = device_obj
        self.dtype = dtype

        self.scale_factor = scale_factor
        self.vae_scale_factor = vae_scale_factor
        self.vae_shift_factor = vae_shift_factor
        self.n_channels = n_channels
        self.height = height
        self.width = width

        self.alphas_f64 = alphas_f64
        self.sigmas_f64 = sigmas_f64
        self.net_timesteps = net_timesteps
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = timesteps
        self._prepare_networks()

    # ------------------------------------------------------------------
    #  Pipeline memory optim and setup for inference
    # ------------------------------------------------------------------

    def _setup_pipeline_optimizations(
        self,
        pipeline: StableDiffusion3Pipeline,
        device: str | torch.device,
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False,
        enable_attention_slicing: bool = False,
        enable_vae_slicing: bool = False,
    ) -> tuple[StableDiffusion3Pipeline, torch.device]:
        
        device_obj = torch.device(device)

        if enable_model_cpu_offload and enable_sequential_cpu_offload:
            enable_sequential_cpu_offload = True
            enable_model_cpu_offload = False

        if enable_model_cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
            if device_obj.type != "cuda":
                raise ValueError("model_cpu_offload requires CUDA")
            gpu_index = device_obj.index if device_obj.index is not None else 0
            pipeline.enable_model_cpu_offload(gpu_id=gpu_index)
            device_obj = torch.device(f"cuda:{gpu_index}")
        elif enable_sequential_cpu_offload and hasattr(pipeline, "enable_sequential_cpu_offload"):
            if device_obj.type != "cuda":
                raise ValueError("sequential_cpu_offload requires CUDA")
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline.to(device_obj)

        if enable_attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if enable_vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()

        return pipeline, device_obj

    def _prepare_networks(self) -> None:
        modules = [
            self.transformer,
            self.vae,
            self.text_encoder,
            self.text_encoder_2,
            self.text_encoder_3,
        ]
        for module in modules:
            module.eval()
            module.requires_grad_(False)

    # ------------------------------------------------------------------
    # Prompt handling
    # ------------------------------------------------------------------

    def set_prompt(
        self,
        prompts: str | Sequence[str],
        negative_prompts: Optional[str | Sequence[str]] = None,
        guidance_scale: float = 7.0,
        n_images_per_prompt: int = 1,
        **_: Any,
    ) -> None:
        if isinstance(prompts, str):
            prompts = [prompts]
        else:
            prompts = [str(p) for p in prompts]
        if len(prompts) == 0:
            prompts = [""]

        if negative_prompts is None:
            negative_prompts: Optional[list[str]] = None
        elif isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)
        else:
            negative_prompts = [str(p) for p in negative_prompts]
            if len(negative_prompts) not in (1, len(prompts)):
                raise ValueError(
                    "Negative prompts must provide either a single value or one per prompt."
                )
            if len(negative_prompts) == 1 and len(prompts) > 1:
                negative_prompts = negative_prompts * len(prompts)
        
        self.guidance_scale = guidance_scale
        self.do_true_cfg = bool(self.guidance_scale > 1.0)

        (
            prompt_embeds,
            neg_prompt_embeds,
            pooled_embeds,
            neg_pooled_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompts,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=self.do_true_cfg,
            num_images_per_prompt=n_images_per_prompt,
            device=self.device,
        )

        if self.do_true_cfg:
            self.prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds])
            self.pooled_embeds = torch.cat([neg_pooled_embeds, pooled_embeds])
        else:
            self.prompt_embeds = prompt_embeds
            self.pooled_embeds = pooled_embeds

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, images: Tensor, generator: torch.Generator) -> Tensor:
        input_dtype = images.dtype
        images = images.to(self.dtype)
        latents = self.vae.encode(images, return_dict=False)[0]
        latents = (latents.sample(generator=generator) - self.vae_shift_factor) * self.vae_scale_factor
        return latents.to(input_dtype).to(self.device)
    
    def decode(self, latents: Tensor) -> Tensor:
        input_dtype = latents.dtype
        latents = (latents / self.vae_scale_factor) + self.vae_shift_factor
        latents = latents.to(self.dtype)
        images = self.vae.decode(latents, return_dict=False)[0]
        return images.to(input_dtype).to(self.device)

    # ------------------------------------------------------------------
    # Predict velocity
    # ------------------------------------------------------------------

    def pred_velocity(self, latents: Tensor, t: Tensor | int) -> Tensor:
        
        latents = latents.to(self.prompt_embeds.dtype)
        if self.do_true_cfg:
            latents = torch.cat([latents] * 2)

        timestep = self._resolve_timestep_tensor(
            t=t,
            batch_size=latents.shape[0],
            device=latents.device,
        )

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=self.prompt_embeds,
            pooled_projections=self.pooled_embeds,
            return_dict=False,
        )[0]

        if self.do_true_cfg:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = self.guidance_scale * noise_pred_cond + (1 - self.guidance_scale) * noise_pred_uncond

        return noise_pred

    # ------------------------------------------------------------------
    # Setup input params and timesteps
    # ------------------------------------------------------------------
    
    def set_image_params(
            self,
            height: int,
            width: int,
        ) -> None:

        self.height = int(height or self.height)
        self.width = int(width or self.width)
        self.latent_height = 2 * (self.height // (self.scale_factor * 2))
        self.latent_width = 2 * (self.width // (self.scale_factor * 2))
        self.latent_shape = (self.n_channels, self.latent_height, self.latent_width)
        self.image_seq_len = (self.latent_height // 2) * (self.latent_width // 2)

    def set_timesteps(
            self,
            n_steps: int
        ) -> None:
        
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        else:
            sigmas = np.linspace(1.0, 1 / n_steps, n_steps)

        if hasattr(self.scheduler.config, "use_dynamic_shifting") and self.scheduler.config.use_dynamic_shifting:
            mu = self._calculate_shift(
                self.image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
        else:
            mu = None
        
        self.scheduler.set_timesteps(num_inference_steps=n_steps, device=self.device, sigmas=sigmas, mu=mu)

        timesteps_desc = self.scheduler.timesteps.to(device=self.device, dtype=torch.float32)
        sigmas_desc = self.scheduler.sigmas.to(device=self.device, dtype=torch.float64)

        if sigmas_desc.shape[0] == timesteps_desc.shape[0] + 1:
            sigmas_desc = sigmas_desc[:-1]
        elif sigmas_desc.shape[0] != timesteps_desc.shape[0]:
            raise ValueError(
                "Scheduler produced mismatched sigma and timestep lengths; expected sigmas to "
                "match timesteps or contain a single terminal entry."
            )
        
        self.net_timesteps = timesteps_desc.flip(0)
        self.sigmas_f64 = sigmas_desc.flip(0)
        self.alphas_f64 = (1.0 - self.sigmas_f64).clamp(min=0.0)
        self.alphas = self.alphas_f64.to(self.device, self.dtype)
        self.sigmas = self.sigmas_f64.to(self.device, self.dtype)
        self.timesteps = torch.arange(n_steps, dtype=torch.int64, device=self.device)
        
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_timestep_tensor(
        self,
        t: Tensor | int,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if isinstance(t, Tensor):
            tensor = t
        elif isinstance(t, int):
            tensor = torch.tensor([t], dtype=torch.int64, device=device)
        else:  # pragma: no cover - defensive
            raise TypeError("Timesteps must be a tensor or integer index.")

        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        elif tensor.ndim != 1:
            raise ValueError("Timesteps tensor must have at most one dimension.")

        if tensor.dtype.is_floating_point:
            timestep = tensor.to(device=device, dtype=torch.float32)
        else:
            indices = tensor.to(device=device, dtype=torch.int64)
            timestep = self.net_timesteps.index_select(0, indices.view(-1))

        if timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)

        return timestep
    
    @staticmethod
    def _calculate_shift(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return float(mu)
    

# ------------------------------------------------------------------
# Stable Diffusion 3 family
# ------------------------------------------------------------------

class SD3MediumDenoiser(StableDiffusion3Denoiser):
    """Stability AI Stable Diffusion 3 medium denoiser wrapper."""
    default_model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
class SD35MediumDenoiser(StableDiffusion3Denoiser):
    """Stable Diffusion 3.5 medium denoiser wrapper."""
    default_model_id = "stabilityai/stable-diffusion-3.5-medium"
class SD35LargeDenoiser(StableDiffusion3Denoiser):
    """Stable Diffusion 3.5 large denoiser wrapper."""
    default_model_id = "stabilityai/stable-diffusion-3.5-large"
class SD35LargeTurboDenoiser(StableDiffusion3Denoiser):
    """Stable Diffusion 3.5 large turbo denoiser wrapper."""
    default_model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
