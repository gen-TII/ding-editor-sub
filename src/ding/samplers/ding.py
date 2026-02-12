"""Implementation of the Ding inpainting sampler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor

from ding.api import BaseSampler, RunArtifacts
from ding.utils import (
    load_and_resize_image,
    load_and_resize_video,
    load_and_resize_audio,
    resolve_image_mask,
    load_video_mask,
    resolve_audio_mask,
    resize_mask_for_latent,
    persist_image_outputs,
    persist_video_outputs,
    persist_audio_outputs,
)


class DingSampler(BaseSampler):
    """DING sampler specialised for latent image and video inpainting."""

    def __init__(
        self,
        denoiser: Any,
        eta_type: str = "square",
        obs_std: float = 0.01,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device or getattr(denoiser, "device", "cpu"))
        self.denoiser = denoiser
        self.eta_type = eta_type
        self.obs_std = obs_std

    def sample_image(
        self,
        image_path: str | Path,
        mask_path: Optional[str | Path],
        out_dir: str | Path,
        steps: int,
        n_samples: int = 1,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        show_progress: bool = False,
        seed: Optional[int] = None,
        save_outputs: bool = True,
        dilate_mask: bool = False,
        **_: Any,
    ) -> RunArtifacts:
        device = self.device
        dtype = getattr(self.denoiser, "dtype", torch.float32)
        n_samples = max(1, int(n_samples))

        out_dir = self._prepare_output_dir(out_dir)

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        reference_image = load_and_resize_image(
            image_path,
            device=device,
            dtype=dtype,
            target_height=target_height,
            target_width=target_width,
        ).unsqueeze(0)

        mask_pixel = resolve_image_mask(
            mask_path=mask_path,
            bbox=bbox,
            image_shape=reference_image.shape[-2:],
            device=device,
            dtype=dtype
        ).unsqueeze(0)

        reference_latent = self.denoiser.encode(reference_image, generator)
        mask_latent = resize_mask_for_latent(
            mask_pixel,
            reference_latent.shape,
            threshold=0.95,
            dilate=dilate_mask,
        )
        reference_latent_batch = reference_latent.repeat(n_samples, 1, 1, 1)
        mask_latent_batch = mask_latent.repeat(n_samples, 1, 1, 1)
        obs_latent_batch = mask_latent_batch * reference_latent_batch

        reconstructions, nfe_count = self._run_sampler(
            obs_latent_batch=obs_latent_batch,
            mask_latent_batch=mask_latent_batch,
            show_progress=show_progress,
            generator=generator,
        )

        files: Dict[str, Any] = {}
        if save_outputs:
            files = persist_image_outputs(
                out_dir=out_dir,
                observation=reference_image * mask_pixel,
                reconstructions=reconstructions,
            )

        metadata = {
            "sampler": "ding",
            "steps": steps,
            "nfe_count": nfe_count,
            "eta_type": self.eta_type,
            "image_path": str(image_path),
            "mask_path": None if mask_path is None else str(mask_path),
            "dilate_mask": dilate_mask,
            "obs_std": self.obs_std,
            "device": device,
            "seed": seed,
            "n_samples": n_samples,
            "target_height": target_height,
            "target_width": target_width,
        }

        if save_outputs:
            metadata_path = out_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))
            files = {**files, "metadata": metadata_path}

        return RunArtifacts(
            reconstructions=reconstructions.detach(),
            output_dir=out_dir,
            metadata=metadata,
            files=files,
        )

    def sample_video(
        self,
        video_path: str | Path,
        mask_path: Optional[str | Path],
        out_dir: str | Path,
        steps: int,
        n_samples: int = 1,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        target_num_frames: Optional[int] = None,
        frame_rate: Optional[int] = None,
        show_progress: bool = False,
        seed: Optional[int] = None,
        save_outputs: bool = True,
        dilate_mask: bool = True,
        **_: Any,
    ) -> RunArtifacts:
        device = self.device
        dtype = getattr(self.denoiser, "dtype", torch.float32)
        n_samples = max(1, int(n_samples))

        out_dir = self._prepare_output_dir(out_dir)

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        reference_video = load_and_resize_video(
            video_path,
            device=device,
            dtype=dtype,
            target_height=target_height,
            target_width=target_width,
            target_num_frames=target_num_frames,
        ).unsqueeze(0)
        
        mask_pixel = load_video_mask(
            mask_path,
            device=device,
            dtype=dtype,
            target_height=int(reference_video.shape[-2]),
            target_width=int(reference_video.shape[-1]),
            target_num_frames=int(reference_video.shape[2]),
        ).unsqueeze(0)

        print(f"mask pixel shape {mask_pixel.shape}, reference_video shape {reference_video.shape}")
        reference_latent = self.denoiser.encode(reference_video, generator)

        mask_latent = resize_mask_for_latent(
            mask_pixel,
            reference_latent.shape,
            threshold=0.95,
            dilate=dilate_mask,
        )
        print(f"mask latent shape {mask_latent.shape}, reference_latent shape {reference_latent.shape}")

        reference_latent_batch = reference_latent.repeat(n_samples, 1, 1, 1, 1)
        mask_latent_batch = mask_latent.repeat(n_samples, 1, 1, 1, 1)
        obs_latent_batch = mask_latent_batch * reference_latent_batch
        
        reconstructions, nfe_count = self._run_sampler(
            obs_latent_batch=obs_latent_batch,
            mask_latent_batch=mask_latent_batch,
            show_progress=show_progress,
            generator=generator,
        )

        files: Dict[str, Any] = {}
        if save_outputs:
            files = persist_video_outputs(
                out_dir=out_dir,
                observation=reference_video * mask_pixel,
                reconstructions=reconstructions,
                frame_rate=frame_rate,
            )

        metadata = {
            "sampler": "ding",
            "mode": "video",
            "steps": steps,
            "nfe_count": nfe_count,
            "eta_type": self.eta_type,
            "video_path": str(video_path),
            "mask_path": None if mask_path is None else str(mask_path),
            "dilate_mask": dilate_mask,
            "obs_std": self.obs_std,
            "device": device,
            "seed": seed,
            "n_samples": n_samples,
            "target_height": target_height,
            "target_width": target_width,
            "target_num_frames": target_num_frames,
            "frame_rate": frame_rate,
        }

        if save_outputs:
            metadata_path = out_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))
            files = {**files, "metadata": metadata_path}

        return RunArtifacts(
            reconstructions=reconstructions.detach(),
            output_dir=out_dir,
            metadata=metadata,
            files=files,
        )

    def sample_audio(
        self,
        audio_path: str | Path,
        mask_path: Optional[str | Path],
        out_dir: str | Path,
        steps: int,
        n_samples: int = 1,
        audio_start_in_s: Optional[int] = None,
        audio_end_in_s: Optional[int] = None,
        bbox: Optional[Tuple[float, float]] = None,
        show_progress: bool = False,
        seed: Optional[int] = None,
        save_outputs: bool = True,
        dilate_mask: bool = False,
        **_: Any,
    ) -> RunArtifacts:
        device = self.device
        dtype = getattr(self.denoiser, "dtype", torch.float32)
        n_samples = max(1, int(n_samples))

        out_dir = self._prepare_output_dir(out_dir)

        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        reference_audio = load_and_resize_audio(
            audio_path,
            device=device,
            dtype=dtype,
            target_length=self.denoiser.audio_vae_length,
            target_sample_rate=self.denoiser.sample_rate,
        ).unsqueeze(0)

        mask_time = resolve_audio_mask(
            mask_path,
            bbox=bbox,
            sample_rate=self.denoiser.sample_rate,
            device=device,
            dtype=dtype,
            target_length=self.denoiser.audio_vae_length,
        ).unsqueeze(0)

        reference_latent = self.denoiser.encode(reference_audio, generator)
        mask_latent = resize_mask_for_latent(
            mask_time,
            reference_latent.shape,
            threshold=0.95,
            dilate=dilate_mask,
        )

        reference_latent_batch = reference_latent.repeat(n_samples, 1, 1)
        mask_latent_batch = mask_latent.repeat(n_samples, 1, 1)
        obs_latent_batch = mask_latent_batch * reference_latent_batch
        print(f"reference_latent_batch shape: {reference_latent_batch.shape}, mask_latent_batch shape: {mask_latent_batch.shape}")

        reconstructions, nfe_count = self._run_sampler(
            obs_latent_batch=obs_latent_batch,
            mask_latent_batch=mask_latent_batch,
            show_progress=show_progress,
            generator=generator,
        )

        files: Dict[str, Any] = {}
        if save_outputs:
            files = persist_audio_outputs(
                out_dir=out_dir,
                observation=reference_audio * mask_time,
                reconstructions=reconstructions,
                sample_rate=self.denoiser.sample_rate,
            )

        metadata = {
            "sampler": "ding",
            "steps": steps,
            "nfe_count": nfe_count,
            "eta_type": self.eta_type,
            "audio_path": str(audio_path),
            "mask_path": None if mask_path is None else str(mask_path),
            "dilate_mask": dilate_mask,
            "obs_std": self.obs_std,
            "device": device,
            "seed": seed,
            "n_samples": n_samples,
            "audio_start_in_s": audio_start_in_s,
            "audio_end_in_s": audio_end_in_s,
        }

        if save_outputs:
            metadata_path = out_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))
            files = {**files, "metadata": metadata_path}

        return RunArtifacts(
            reconstructions=reconstructions.detach(),
            output_dir=out_dir,
            metadata=metadata,
            files=files,
        )
        
    # ---------------------------------------------------------------------
    # Sampler runners
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def _run_sampler(
        self,
        obs_latent_batch: torch.Tensor,
        mask_latent_batch: torch.Tensor,
        show_progress: bool,
        generator: torch.Generator,
    ) -> torch.Tensor:
        
        nfe_count = 0
        timesteps = self.denoiser.timesteps.flip(0)
        alphas, sigmas, alphas_f64, sigmas_f64 = self._get_scheduler_tensors(self.denoiser)

        x_t = randn_tensor(obs_latent_batch.shape, generator=generator, device=obs_latent_batch.device, dtype=obs_latent_batch.dtype)
        if hasattr(self.denoiser.scheduler, "init_noise_sigma"):
            x_t = x_t * self.denoiser.scheduler.init_noise_sigma

        progress_bar = tqdm(enumerate(timesteps[:-2]), desc="DING", disable=not show_progress)
        for step_idx, t_idx_tensor in progress_bar:
            t_idx = int(t_idx_tensor.item())
            t_prev_tensor = timesteps[step_idx + 1]
            t_prev_idx = int(t_prev_tensor.item())
        
            alpha_t_prev = alphas[t_prev_idx]
            sigma_t_prev = sigmas[t_prev_idx]
            
            eta = self._get_eta(
                t_idx=t_idx,
                t_prev_idx=t_prev_idx,
                alphas=alphas_f64,
                sigmas=sigmas_f64,
                dtype=x_t.dtype,
                eta_type=self.eta_type,
            )
        
            pred_x0 = self.denoiser.pred_x0(x_t, t_idx_tensor)
            nfe_count += 1
            mean_prior, std_prior = self._bridge_kernel(
                x_t=x_t,
                x_0=pred_x0,
                t=t_idx,
                t_prev=t_prev_idx,
                alphas=alphas_f64,
                sigmas=sigmas_f64,
                dtype=x_t.dtype,
                eta=eta,
            )
            z_s = mean_prior + std_prior * randn_tensor(mean_prior.shape, generator=generator, device=mean_prior.device, dtype=mean_prior.dtype)

            # For VE/EDM-style models (e.g. Stable Audio Open), alpha(t)=1 and x_t = x0 + sigma(t) * eps.
            # In that case we want the unit-variance noise term eps (our denoiser exposes it as `pred_x1`).
            if torch.isclose(alpha_t_prev, torch.tensor(1.0, device=alpha_t_prev.device, dtype=alpha_t_prev.dtype)):
                noise_pred = self.denoiser.pred_x1(z_s, t_prev_tensor)
                nfe_count += 1
            else:
                noise_pred = z_s + alpha_t_prev * self.denoiser.pred_velocity(z_s, t_prev_tensor)
                nfe_count += 1

            x_t = self._sample_posterior(
                obs=alpha_t_prev * obs_latent_batch
                + sigma_t_prev * (mask_latent_batch * noise_pred),
                mask=mask_latent_batch,
                obs_std=self.obs_std * alpha_t_prev,
                mean_prior=mean_prior,
                std_prior=std_prior,
                generator=generator,
            )
        
        last_t = timesteps[-2]
        x_0 = self.denoiser.pred_x0(x_t, last_t)
        nfe_count += 1
        decoded = self.denoiser.decode(x_0)
        return decoded.clamp(-1.0, 1.0), nfe_count

    # ---------------------------------------------------------------------
    # Internal sampler helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _get_scheduler_tensors(
        denoiser: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        alphas = getattr(denoiser, "alphas")
        sigmas = getattr(denoiser, "sigmas")
        alphas_f64 = getattr(denoiser, "alphas_f64")
        sigmas_f64 = getattr(denoiser, "sigmas_f64")

        return alphas, sigmas, alphas_f64, sigmas_f64

    @staticmethod
    def _get_eta(
        t_idx: int,
        t_prev_idx: int,
        alphas: torch.Tensor,
        sigmas: torch.Tensor,
        dtype: torch.dtype,
        eta_type: str,
    ) -> torch.Tensor:
        alpha_t = alphas[t_idx]
        alpha_t_prev = alphas[t_prev_idx]
        sigma_t = sigmas[t_idx]
        sigma_t_prev = sigmas[t_prev_idx]

        a_t_t_prev = alpha_t / alpha_t_prev
        sig_t_prev_t = sigma_t_prev / sigma_t

        if eta_type == "default":
            eta = ((1 - alpha_t_prev) / (1 - (a_t_t_prev * sig_t_prev_t) ** 2)).sqrt()
        elif eta_type == "square":
            eta = (
                (1 - alpha_t_prev).square() / (1 - (a_t_t_prev * sig_t_prev_t) ** 2)
            ).sqrt()
        elif eta_type == "max":
            eta = torch.minimum(
                1
                / ((1 - (a_t_t_prev * sig_t_prev_t) ** 2).sqrt()),
                torch.tensor(1.0, device=alpha_t.device),
            )
        elif eta_type == "ddpm":
            eta = torch.tensor(1.0, device=alpha_t.device)
        elif eta_type == "ddim":
            eta = torch.tensor(1e-2, device=alpha_t.device)
        else:
            raise ValueError(f"Unknown eta_type '{eta_type}'.")

        return eta.to(dtype)

    @staticmethod
    def _bridge_kernel(
        x_t: torch.Tensor,
        x_0: torch.Tensor,
        t: int,
        t_prev: int,
        alphas: torch.Tensor,
        sigmas: torch.Tensor,
        dtype: torch.dtype,
        eta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_t = alphas[t]
        alpha_t_prev = alphas[t_prev]
        sigma_t = sigmas[t]
        sigma_t_prev = sigmas[t_prev]

        a_t_t_prev = alpha_t / alpha_t_prev
        sig_t_prev_t = sigma_t_prev / sigma_t

        var = sigma_t_prev**2 * (1 - (a_t_t_prev * sig_t_prev_t) ** 2)
        var = (eta**2) * var

        coef_x_t = (sig_t_prev_t**2 - var / sigma_t**2).sqrt()
        coef_x_0 = alpha_t_prev - alpha_t * coef_x_t

        mean = coef_x_t.to(dtype) * x_t + coef_x_0.to(dtype) * x_0
        std = var.sqrt().to(dtype)
        return mean, std

    @staticmethod
    def _sample_posterior(
        obs: torch.Tensor,
        mask: torch.Tensor,
        obs_std: float,
        mean_prior: torch.Tensor,
        std_prior: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        eps = 1e-6
        std_prior_sq = std_prior.square().clamp_min(eps)
        obs_tensor = torch.as_tensor(obs_std, device=mean_prior.device, dtype=mean_prior.dtype)
        obs_var = obs_tensor.square()

        # mask is broadcast-compatible with mean_prior; ensure float
        mask = mask.to(mean_prior.dtype)

        precision_post = (mask.square() / (obs_var + eps)) + 1.0 / std_prior_sq
        precision_post = precision_post.clamp_min(eps)
        cov = 1.0 / precision_post
        unscaled_mean = (mask.square() * obs) / (obs_var + eps) + mean_prior / std_prior_sq

        posterior_mean = cov * unscaled_mean
        noise = randn_tensor(mean_prior.shape, generator=generator, device=mean_prior.device, dtype=mean_prior.dtype)
        return posterior_mean + cov.sqrt() * noise
