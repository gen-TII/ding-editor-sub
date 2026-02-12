"""Implementation of the BlendedDiffusion inpainting sampler."""

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


class BlendedDiffusionSampler(BaseSampler):
    """BlendedDiffusion sampler specialised for latent image and video inpainting."""

    def __init__(
        self,
        denoiser: Any,
        blending_fraction: float = 0.0,
        eta: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device or getattr(denoiser, "device", "cpu"))
        self.denoiser = denoiser
        self.blending_fraction = blending_fraction
        self.eta = eta

    def sample_image(
        self,
        image_path: str | Path,
        mask_path: Optional[str | Path],
        out_dir: str | Path,
        steps: Optional[int] = None,
        target_height: Optional[int] = None,
        target_width: Optional[int] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        n_samples: int = 1,
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
            dtype=dtype,
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
            "sampler": "BlendedDiffusion",
            "steps": steps,
            "nfe_count": nfe_count,
            "device": device,
            "seed": seed,
            "dilate_mask": dilate_mask,
            "n_samples": n_samples,
            "target_height": target_height,
            "target_width": target_width,
            "image_path": str(image_path),
            "mask_path": None if mask_path is None else str(mask_path),
        }

        if save_outputs:
            meta_path = out_dir / "metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2))
            files = {**files, "metadata": meta_path}

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

        reference_latent = self.denoiser.encode(reference_video, generator)
        mask_latent = resize_mask_for_latent(
            mask_pixel,
            reference_latent.shape,
            threshold=0.95,
            dilate=dilate_mask,
        )
        
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
                frame_rate=frame_rate
            )

        metadata = {
            "sampler": "BlendedDiffusion",
            "mode": "video",
            "steps": steps,
            "nfe_count": nfe_count,
            "video_path": str(video_path),
            "mask_path": None if mask_path is None else str(mask_path),
            "dilate_mask": dilate_mask,
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
            device=device,
            dtype=dtype,
            sample_rate=self.denoiser.sample_rate,
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
            "sampler": "BlendedDiffusion",
            "steps": steps,
            "nfe_count": nfe_count,
            "device": device,
            "seed": seed,
            "dilate_mask": dilate_mask,
            "n_samples": n_samples,
            "audio_start_in_s": audio_start_in_s,
            "audio_end_in_s": audio_end_in_s,
            "audio_path": str(audio_path),
            "mask_path": None if mask_path is None else str(mask_path),
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
            
    # ------------------------------------------------------------------
    # Sampler runners
    # ------------------------------------------------------------------

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

        z = randn_tensor(obs_latent_batch.shape, generator=generator, device=obs_latent_batch.device, dtype=obs_latent_batch.dtype)
        blend_start = int(self.blending_fraction * (len(timesteps) - 1))
        blend_start = max(0, min(blend_start, len(timesteps) - 2))

        progress_bar = tqdm(enumerate(timesteps[blend_start:-1], start=blend_start), desc="BlendedDiffusion", disable=not show_progress)
        for step_idx, t_idx in progress_bar:
            t_prev_idx = timesteps[step_idx + 1]

            x0_fg = self.denoiser.pred_x0(z, t_idx)
            nfe_count += 1

            z_fg, _ = self._bridge_kernel(
                x_t=z,
                x_0=x0_fg,
                t=t_idx,
                t_prev=t_prev_idx,
                alphas=alphas_f64,
                sigmas=sigmas_f64,
                dtype=z.dtype,
                eta=self.eta,
            )

            # NOTE in the official implementation the authors noise up to t_idx
            # https://github.com/omriav/blended-latent-diffusion/blob/09565c5d9ca260de1fd22260784eb744874f9004/scripts/text_editing_SD2.py#L140-L143
            # but this actually results in tiny artifacts in the reconstruction
            # here we use `t_prev_idx` to match the noise level in `z_fg`
            alpha = self.denoiser.alphas[t_prev_idx]
            sigma = self.denoiser.sigmas[t_prev_idx]
            z_bg = alpha * obs_latent_batch + sigma * randn_tensor(obs_latent_batch.shape, generator=generator, device=obs_latent_batch.device, dtype=obs_latent_batch.dtype)
            z = (1 - mask_latent_batch) * z_fg + mask_latent_batch * z_bg

        decoded = self.denoiser.decode(z)
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