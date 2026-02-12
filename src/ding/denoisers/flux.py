"""Flux image denoiser wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import Tensor
import numpy as np

from diffusers.utils.torch_utils import randn_tensor
from .base import BaseDenoiser
from diffusers import FluxPipeline


class FluxDenoiser(BaseDenoiser):
    """Wrapper around the Flux image generation pipeline."""

    default_model_id: str = "black-forest-labs/FLUX.1-dev"

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
        pipeline = FluxPipeline.from_pretrained(model_id, **load_kwargs)

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
        transformer = pipeline.transformer
        vae = pipeline.vae
        scheduler = pipeline.scheduler

        scale_factor = pipeline.vae_scale_factor
        vae_scale_factor = vae.config.scaling_factor   
        vae_shift_factor = vae.config.shift_factor
        height = width = pipeline.default_sample_size * scale_factor
        n_channels = transformer.config.in_channels // 4

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
        pipeline: FluxPipeline,
        device: str | torch.device,
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False,
        enable_attention_slicing: bool = False,
        enable_vae_slicing: bool = False,
    ) -> tuple[FluxPipeline, torch.device]:
        
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
        ]
        for module in modules:
            module.eval()
            module.requires_grad_(False)

    # ------------------------------------------------------------------
    # Prompt and guidance handling
    # ------------------------------------------------------------------

    def set_prompt(
        self,
        prompts: str | Sequence[str],
        negative_prompts: Optional[str | Sequence[str]] = None,
        guidance_scale: float = 7.0,
        n_images_per_prompt: int = 1,
        true_cfg_scale: float | None = None,
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
        self.guidance = torch.tensor([self.guidance_scale], dtype=torch.float32, device=self.device)
        self.guidance = self._guidance_tensor(batch_size=n_images_per_prompt*len(prompts))
        self.true_cfg_scale = true_cfg_scale
        self.do_true_cfg = bool(self.true_cfg_scale > 1.0 and negative_prompts is not None)

        prompt_embeds, pooled_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            num_images_per_prompt=n_images_per_prompt,
            device=self.device,
            max_sequence_length=512,
        )
        self.prompt_embeds = prompt_embeds
        self.pooled_embeds = pooled_embeds
        self.text_ids = text_ids

        if self.do_true_cfg:
            neg_prompt_embeds, neg_pooled_embeds, neg_text_ids = self.pipeline.encode_prompt(
                prompt=negative_prompts,
                prompt_2=None,
                num_images_per_prompt=n_images_per_prompt,
                device=self.device,
            )
            self.neg_prompt_embeds = neg_prompt_embeds
            self.neg_pooled_embeds = neg_pooled_embeds
            self.neg_text_ids = neg_text_ids
        else:
            self.neg_prompt_embeds = None
            self.neg_pooled_embeds = None
            self.neg_text_ids = None

    def _guidance_tensor(self, batch_size: int) -> Tensor | None:
        if self.guidance is None:
            return None
        if self.guidance.ndim == 0:
            return self.guidance.reshape(1).expand(batch_size)
        if self.guidance.shape[0] == batch_size:
            return self.guidance
        return self.guidance.expand(batch_size)

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
        
        packed_latents = self._pack_latents(latents)
        packed_latents = packed_latents.to(self.prompt_embeds.dtype)
        
        timestep = self._resolve_timestep_tensor(
            t=t,
            batch_size=packed_latents.shape[0],
            device=latents.device,
        )

        noise_pred = self.transformer(
            hidden_states=packed_latents,
            timestep= timestep / self.num_train_timesteps,
            guidance=self.guidance,
            pooled_projections=self.pooled_embeds,
            encoder_hidden_states=self.prompt_embeds,
            txt_ids=self.text_ids,
            img_ids=self.latent_image_ids,
            return_dict=False,
        )[0]

        if self.do_true_cfg:
            noise_pred_uncond = self.transformer(
                hidden_states=packed_latents,
                timestep=timestep / self.num_train_timesteps,
                guidance=self.guidance,
                pooled_projections=self.neg_pooled_embeds,
                encoder_hidden_states=self.neg_prompt_embeds,
                txt_ids=self.neg_text_ids,
                img_ids=self.latent_image_ids,
                return_dict=False,
            )[0]
            noise_pred = self.true_cfg_scale * noise_pred + (1 - self.true_cfg_scale) * noise_pred_uncond

        noise_pred = self._unpack_latents(noise_pred)
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

        half_latent_size = (self.latent_height // 2, self.latent_width // 2)
        latent_image_ids = torch.zeros(
            (half_latent_size[0], half_latent_size[1], 3),
            dtype=torch.float32,
            device=self.device,
        )
        latent_image_ids[..., 1] += torch.arange(half_latent_size[0], device=self.device, dtype=torch.float32)[:, None]
        latent_image_ids[..., 2] += torch.arange(half_latent_size[1], device=self.device, dtype=torch.float32)[None, :]
        self.latent_image_ids = latent_image_ids.view(-1, 3)
  
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

    def _pack_latents(self, latents: Tensor) -> Tensor:
        batch_size = latents.shape[0]
        latents = latents.view(batch_size, self.latent_shape[0], self.latent_shape[1] // 2, 2, self.latent_shape[2] // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        num_patches = (self.latent_shape[1] // 2) * (self.latent_shape[2] // 2)
        channels = self.latent_shape[0] * 4
        latents = latents.reshape(batch_size, num_patches, channels)
        return latents

    def _unpack_latents(self, latents: Tensor) -> Tensor:
        batch_size, num_patches, channels = latents.shape
        latents = latents.view(batch_size, self.latent_height // 2, self.latent_width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), self.latent_height, self.latent_width)
        return latents
    
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
    # Sampling pipelines
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_manual_wrapper(
        self,
        generator: torch.Generator,
        prompt: str,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 3.5,
        true_cfg_scale: float = 1.0,
        n_images_per_prompt: int = 1,
        num_inference_steps: int = 28,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tensor:
        """Manual Flux loop that reuses this wrapper's helpers."""

        # OK - Initial setup
        self.pipeline.set_progress_bar_config(disable=False)
        batch_size = 1
        device = self.pipeline._execution_device

        # OK - Set image params & prompt
        self.set_image_params(
            height=height,
            width=width,
        )
        self.set_prompt(
            prompts=prompt,
            negative_prompts=negative_prompt,
            guidance_scale=guidance_scale,
            n_images_per_prompt=n_images_per_prompt,
            true_cfg_scale=true_cfg_scale,
        )

        # OK - Prepare timesteps
        self.set_timesteps(num_inference_steps)
        timesteps = self.net_timesteps.flip(0)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # OK - Init latents
        shape = (batch_size, *self.latent_shape)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=self.dtype)

        # OK - Denosing loop
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = self.pred_velocity(latents, t)
                latents_dtype = latents.dtype
                denoised_latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents = denoised_latents
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()       

        # OK - Decode latents
        decoded = self.decode(latents)
        image = self.pipeline.image_processor.postprocess(decoded, output_type="pt")
        if isinstance(image, list):
            image = image[0]
        return image.to(self.device, self.dtype)


    @torch.no_grad()
    def generate_pipeline_baseline(
        self,
        generator: torch.Generator,
        prompt: str,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 3.5,
        true_cfg_scale: float = 1.0,
        n_images_per_prompt: int = 1,
        num_inference_steps: int = 28,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tensor:
        """Run the official diffusers Flux pipeline loop and return the decoded tensor."""

        pipe = self.pipeline
        pipe.set_progress_bar_config(disable=False)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            num_images_per_prompt=n_images_per_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator,
            output_type="pt",
            return_dict=True,
            max_sequence_length=512,
        )
        image = result.images
        if isinstance(image, list):
            image = image[0]
        return image.to(self.device, self.dtype)