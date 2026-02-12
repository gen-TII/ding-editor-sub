"""Implementation of the FlowChef inpainting sampler."""

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

class FlowChefSampler(BaseSampler):
    """FlowChef sampler specialised for latent image and video inpainting."""

    def __init__(
        self,
        denoiser: Any,
        step_size: float = 0.9,
        dc_iters: int = 10,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device or getattr(denoiser, "device", "cpu"))
        self.denoiser = denoiser
        self.step_size = step_size
        self.dc_iters = dc_iters

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
            "sampler": "FlowChef",
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
            "sampler": "FlowChef",
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
            "sampler": "FlowChef",
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
        dtype = getattr(self.denoiser, "dtype", torch.float32)
        timesteps = self.denoiser.timesteps.flip(0)

        z = randn_tensor(obs_latent_batch.shape, generator=generator, device=obs_latent_batch.device, dtype=obs_latent_batch.dtype)

        progress_bar = tqdm(enumerate(timesteps[:-1]), desc="FlowChef", disable=not show_progress)
        for step_idx, t_idx in progress_bar:

            t_idx = timesteps[step_idx]
            t_prev = timesteps[step_idx + 1]
            sigma = self.denoiser.sigmas[t_idx].to(dtype)

            sigma_prev = self.denoiser.sigmas[t_prev].to(dtype)

            alpha = 1.0 - sigma

            # v-pred (+CFG) at step i
            with torch.no_grad():
                v_cfg = self.denoiser.pred_velocity(z, t_idx)
                nfe_count += 1
                z0t = z - sigma * v_cfg  # clean latent
                z1t = z + alpha * v_cfg  # noise latent

            # ---- data-consistency gradient ----
            z_dc = z0t.detach().clone()
            for _ in range(self.dc_iters):
                z_dc = z_dc.requires_grad_(True)               
                diff = mask_latent_batch * z_dc - obs_latent_batch

                loss = torch.linalg.norm(diff.view(1, -1))

                ### In the main repo, they didn’t define the loss as a squared norm but apply pseudo inverse

                grad = torch.autograd.grad(
                    loss, z_dc, retain_graph=False, create_graph=False
                )[0]
                z_dc = (z_dc - self.step_size * grad).detach()

            # ---- FlowChef update ----
            z = z_dc + sigma_prev * (z1t - z0t)

        decoded = self.denoiser.decode(z)
        return decoded.clamp(-1.0, 1.0), nfe_count