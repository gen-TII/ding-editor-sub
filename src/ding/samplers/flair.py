"""Implementation of the Flair inpainting sampler."""

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


class FlairSampler(BaseSampler):
    """FLAIR sampler specialised for latent image and video inpainting."""

    def __init__(
        self,
        denoiser: Any,
        device: Optional[str] = None,
        regularizer_weight: float = 0.5,
        llh_weight: float = 1.0,
        lr_reg: float = 1.0,
        lr_llh: float = 0.1,
        n_likelihood_steps: int = 15,
        early_stopping: float = 1e-4,
        stopping_threshold: float = 0.18,
    ) -> None:
        super().__init__(device=device or getattr(denoiser, "device", "cpu"))
        self.denoiser = denoiser
        self.regularizer_weight = float(regularizer_weight)
        self.llh_weight = float(llh_weight)
        self.lr_reg = float(lr_reg)
        self.lr_llh = float(lr_llh)
        self.n_likelihood_steps = int(n_likelihood_steps)
        self.early_stopping = float(early_stopping)
        self.stopping_threshold = float(stopping_threshold)

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
            "sampler": "flair",
            "steps": steps,
            "nfe_count": nfe_count,
            "device": device,
            "seed": seed,
            "dilate_mask": dilate_mask,
            "n_samples": n_samples,
            "target_height": target_height,
            "target_width": target_width,
            "regularizer_weight": self.regularizer_weight,
            "llh_weight": self.llh_weight,
            "lr_reg": self.lr_reg,
            "lr_llh": self.lr_llh,
            "n_likelihood_steps": self.n_likelihood_steps,
            "early_stopping": self.early_stopping,
            "stopping_threshold": self.stopping_threshold,
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
            "sampler": "flair",
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
            "regularizer_weight": self.regularizer_weight,
            "llh_weight": self.llh_weight,
            "lr_reg": self.lr_reg,
            "lr_llh": self.lr_llh,
            "n_likelihood_steps": self.n_likelihood_steps,
            "early_stopping": self.early_stopping,
            "stopping_threshold": self.stopping_threshold,
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
            "sampler": "flair",
            "steps": steps,
            "nfe_count": nfe_count,
            "device": device,
            "seed": seed,
            "dilate_mask": dilate_mask,
            "n_samples": n_samples,
            "audio_start_in_s": audio_start_in_s,
            "audio_end_in_s": audio_end_in_s,
            "regularizer_weight": self.regularizer_weight,
            "llh_weight": self.llh_weight,
            "lr_reg": self.lr_reg,
            "lr_llh": self.lr_llh,
            "n_likelihood_steps": self.n_likelihood_steps,
            "early_stopping": self.early_stopping,
            "stopping_threshold": self.stopping_threshold,
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
        dtype = getattr(self.denoiser, "dtype", torch.float32)
        timesteps = self.denoiser.timesteps.flip(0)

        latent_mu = obs_latent_batch.clone().detach().requires_grad_(True)
        optimizer_reg = torch.optim.SGD(params=[latent_mu], lr=self.lr_reg)
        optimizer_llh = torch.optim.SGD(params=[latent_mu], lr=self.lr_llh)

        eps_hat = randn_tensor(
            latent_mu.shape,
            device=latent_mu.device,
            dtype=latent_mu.dtype,
            generator=generator,
        )

        progress_bar = tqdm(enumerate(timesteps[2:]), desc="FLAIR", disable=not show_progress)
        for step_idx, t_idx_tensor in progress_bar:
            t_value = self.denoiser.sigmas[t_idx_tensor].to(device=self.device, dtype=latent_mu.dtype)
            if t_value < self.stopping_threshold:
                break

            alpha = (1 - t_value).to(latent_mu.dtype)
            current_reg_w = self.regularizer_weight
            current_llh_w = self.llh_weight * current_reg_w

            with torch.no_grad():
                x_t = (1 - t_value) * latent_mu + t_value * eps_hat
                v_pred = self.denoiser.pred_velocity(x_t.to(dtype), t_idx_tensor)
                nfe_count += 1
                u_t = eps_hat - latent_mu
                x_1 = x_t + (1 - t_value) * v_pred.to(x_t.dtype)
                noise = randn_tensor(
                    x_1.shape,
                    device=x_1.device,
                    dtype=x_1.dtype,
                    generator=generator,
                )
                eps_hat = alpha * x_1 + (1 - alpha**2).sqrt() * noise

            reg_term = current_reg_w * ((v_pred - u_t) * latent_mu).to(torch.float32).sum()
            reg_term = reg_term.to(latent_mu.dtype)
            reg_term.backward()
            optimizer_reg.step()
            optimizer_reg.zero_grad(set_to_none=True)

            for _ in range(self.n_likelihood_steps):
                diff = mask_latent_batch * latent_mu - obs_latent_batch
                loss_data = current_llh_w * diff.pow(2).sum()

                if loss_data < self.early_stopping * obs_latent_batch.numel():
                    optimizer_llh.zero_grad(set_to_none=True)
                    break

                loss_for_backward = loss_data.to(latent_mu.dtype)
                loss_for_backward.backward()
                optimizer_llh.step()
                optimizer_llh.zero_grad(set_to_none=True)

        latent_mu = latent_mu.detach().to(dtype)       
        decoded = self.denoiser.decode(latent_mu)
        return decoded.clamp(-1.0, 1.0), nfe_count
