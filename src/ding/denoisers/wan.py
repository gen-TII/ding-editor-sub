"""Wan2.1 video denoiser wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import Tensor
import numpy as np

from diffusers.utils.torch_utils import randn_tensor
from .base import BaseDenoiser
from diffusers import AutoencoderKLWan, WanPipeline


class Wan21Denoiser(BaseDenoiser):
    """Wrapper around the Wan2.1 video generation pipeline."""

    default_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    def __init__(
        self,
        device: str = "cuda:0",
        dtype: torch.dtype | str = torch.bfloat16,
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
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
        pipeline = WanPipeline.from_pretrained(model_id, vae=vae, **load_kwargs)

        pipeline, device_obj = self._setup_pipeline_optimizations(
            pipeline=pipeline,
            device=device,
            enable_model_cpu_offload=model_cpu_offload,
            enable_sequential_cpu_offload=sequential_cpu_offload,
            enable_attention_slicing=attention_slicing,
            enable_vae_slicing=vae_slicing,
        )

        text_encoder = pipeline.text_encoder
        transformer = pipeline.transformer
        vae = pipeline.vae
        scheduler = pipeline.scheduler

        vae_spatial_compression = vae.config.scale_factor_spatial
        vae_temporal_compression = vae.config.scale_factor_temporal
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
        self.scheduler = scheduler
        self.transformer = transformer
        self.vae = vae

        self.device = device_obj
        self.dtype = dtype

        self.n_channels = n_channels
        self.vae_spatial_compression = vae_spatial_compression
        self.vae_temporal_compression = vae_temporal_compression

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
        pipeline: WanPipeline,
        device: str | torch.device,
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False,
        enable_attention_slicing: bool = False,
        enable_vae_slicing: bool = False,
    ) -> tuple[WanPipeline, torch.device]:
        
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
        guidance_scale: float = 3.0,
        n_videos_per_prompt: int = 1,
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
        
        self.prompt_embeds, self.negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompts,
            negative_prompt=negative_prompts,
            do_classifier_free_guidance=self.do_true_cfg,
            num_videos_per_prompt=n_videos_per_prompt,
            device=self.device,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, videos: Tensor, generator: torch.Generator) -> Tensor:
        input_dtype = videos.dtype
        videos = videos.to(self.dtype)
        latents = self.vae.encode(videos, return_dict=False)[0]
        latents = self._normalize_latents(
            latents.sample(generator=generator),
            self.vae.config.latents_mean,
            self.vae.config.latents_std,
            self.vae.config.z_dim,
        )
        return latents.to(input_dtype).to(self.device)

    def decode(self, latents: Tensor) -> Tensor:
        input_dtype = latents.dtype
        latents = latents.to(self.dtype)
        latents = self._denormalize_latents(
            latents,
            self.vae.config.latents_mean,
            self.vae.config.latents_std,
            self.vae.config.z_dim,
        )
        videos = self.vae.decode(latents, return_dict=False)[0]
        return videos.to(input_dtype).to(self.device)

    @staticmethod
    def _normalize_latents(
        latents: Tensor,
        latents_mean: list,
        latents_std: list,
        z_dim: int,
    ) -> Tensor:
        latents_mean = (
            torch.tensor(latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(latents_std)
            .view(1, z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = (latents - latents_mean) / latents_std
        return latents

    @staticmethod
    def _denormalize_latents(
        latents: Tensor,
        latents_mean: list,
        latents_std: list,
        z_dim: int,
    ) -> Tensor:
        latents_mean = (
            torch.tensor(latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(latents_std).view(1, z_dim, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        return latents
    
    # ------------------------------------------------------------------
    # Predict velocity
    # ------------------------------------------------------------------

    def pred_velocity(self, latents: Tensor, t: Tensor | int) -> Tensor:  

        latents = latents.to(self.prompt_embeds.dtype)
        timestep = self._resolve_timestep_tensor(
            t=t,
            batch_size=latents.shape[0],
            device=latents.device,
        ).unsqueeze(-1)

        noise_pred = self.transformer(
            hidden_states=latents,
            encoder_hidden_states=self.prompt_embeds,
            timestep=timestep,
            return_dict=False,
        )[0]

        if self.do_true_cfg:
            noise_uncond = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=self.negative_prompt_embeds,
                timestep=timestep,
                return_dict=False,
            )[0]
            noise_pred = self.guidance_scale * noise_pred + (1 - self.guidance_scale) * noise_uncond
        
        return noise_pred

    # ------------------------------------------------------------------
    # Setup input params and timesteps
    # ------------------------------------------------------------------

    def set_video_params(
        self,
        batch_size: int | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        **_: Any,
    ) -> None:
        
        self.height = int(height)
        self.width = int(width)
        self.num_frames = int(num_frames)
    
        if self.height % self.vae_spatial_compression != 0:
            raise ValueError(f"Height must be divisible by the VAE spatial compression ratio {self.vae_spatial_compression}.")
        if self.width % self.vae_spatial_compression != 0:
            raise ValueError(f"Width must be divisible by the VAE spatial compression ratio {self.vae_spatial_compression}.")
        if (self.num_frames - 1) % self.vae_temporal_compression != 0:
            raise ValueError(
                f"Number of frames minus one must be divisible by the VAE temporal compression ratio {self.vae_temporal_compression}."
            )

        latent_height = self.height // self.vae_spatial_compression
        latent_width = self.width // self.vae_spatial_compression
        latent_frames = ((self.num_frames - 1) // self.vae_temporal_compression) + 1
        self.latent_shape = (batch_size, self.n_channels, latent_frames, latent_height, latent_width)
        self.video_sequence_length = self.latent_shape[2] * self.latent_shape[3] * self.latent_shape[4]
        
    def set_timesteps(
            self,
            n_steps: int
        ) -> None:
        
        self.scheduler.set_timesteps(num_inference_steps=n_steps, device=self.device)
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
    
    # ------------------------------------------------------------------
    # Sampling pipelines
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_manual_wrapper(
        self,
        generator: torch.Generator,
        prompt: str,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 3.0,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
    ) -> Tensor:
        """Manual Wan2.1 loop that reuses this wrapper's prompt/text/latents helpers."""

        # OK - Initial setup
        self.pipeline.set_progress_bar_config(disable=False)
        batch_size = 1
        device = self.pipeline._execution_device
        self._guidance_scale = guidance_scale

        # OK - Set video params & prompt
        self.set_video_params(
            batch_size=batch_size,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        self.set_prompt(
            prompts=prompt,
            negative_prompts=negative_prompt,
            guidance_scale=guidance_scale,
            n_videos_per_prompt=1,
        )

        # OK - Prepare timesteps
        self.set_timesteps(num_inference_steps)
        timesteps = self.net_timesteps.flip(0)
        print(f"Timesteps {timesteps} has shape: {timesteps.shape} and dtype: {timesteps.dtype}")
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # OK - Init latents
        latents = randn_tensor(self.latent_shape, generator=generator, device=device, dtype=self.dtype)
        print(f"Latents shape: {latents.shape} and dtype: {self.dtype}")

        # OK - Denosing loop
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = self.pred_velocity(latents, t)
                denoised_latents = self.scheduler.step(
                    noise_pred,
                    t, latents,
                    return_dict=False
                )[0]
                latents = denoised_latents
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()       

        # OK - Decode latents
        video = self.decode(latents)
        video = self.pipeline.video_processor.postprocess_video(video, output_type="pt")
        if isinstance(video, list):
            video = video[0]
        return video.to(device, self.pipeline.dtype)
    
    @torch.no_grad()
    def generate_pipeline_baseline(
        self,
        generator: torch.Generator,
        prompt: str,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
    ) -> Tensor:
        """Run the official diffusers Wan2.1 pipeline loop and return the decoded video tensor."""
 
        self.pipeline.set_progress_bar_config(disable=False)
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            output_type="pt",
            return_dict=True,
            generator=generator,
        )
        video = result.frames
        if isinstance(video, list):
            video = video[0]
        return video.to(self.device, self.dtype)