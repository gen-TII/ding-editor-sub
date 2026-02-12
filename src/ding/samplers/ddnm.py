"""Implementation of the DDNM inpainting sampler."""

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


class DDNMSampler(BaseSampler):
    """DDNM sampler specialised for latent image and video inpainting."""

    def __init__(
        self,
        denoiser: Any,
        eta: float = 0.85,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device or getattr(denoiser, "device", "cpu"))
        self.denoiser = denoiser
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
            "sampler": "DDNM",
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
            "sampler": "DDNM",
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
            "sampler": "DDNM",
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
    
        x = randn_tensor(obs_latent_batch.shape, generator=generator, device=obs_latent_batch.device, dtype=obs_latent_batch.dtype)
        device=obs_latent_batch.device

        # setup iteration variables
        skip = len(self.denoiser.alphas)// len(timesteps)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = self._get_schedule_jump(len(timesteps), 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))

        progress_bar = tqdm(enumerate(time_pairs, start=1), desc="DDNM", disable=not show_progress)
        for step_idx, (i, j) in progress_bar:
            i, j = i * skip, j * skip
            # NOTE official implemen sets j to -1 when j < 0 (equivalently, i is 0)
            # this throws an error with cuda as j is timestep feed to the network
            if i == 0:
                break

            # ---
            t_idx = torch.tensor(i, dtype=torch.int32)
            t_prev_idx = torch.tensor(j, dtype=torch.int32)

            alpha_t, sigma_t = self.denoiser.alphas[i], self.denoiser.sigmas[i]
            alpha_t_prev, sigma_t_prev = self.denoiser.alphas[j], self.denoiser.sigmas[j]
            # ----

            if j < i:  # normal sampling
                xt = xs[-1].to(device)
                x0_t = self.denoiser.pred_x0(xt, t_idx)
                nfe_count += 1
                et = (xt - alpha_t * x0_t) / sigma_t
                x0_t_hat = x0_t - mask_latent_batch * (mask_latent_batch * x0_t - obs_latent_batch)
                c1 = sigma_t_prev * self.eta
                c2 = sigma_t_prev * ((1 - self.eta**2) ** 0.5)
                xt_next = alpha_t_prev * x0_t_hat + c1 * randn_tensor(x0_t.shape, generator=generator, device=x0_t.device, dtype=x0_t.dtype) + c2 * et
                x0_preds.append(x0_t.to("cpu"))
                xs.append(xt_next.to("cpu"))

            else:  # time-travel back
                x0_t = x0_preds[-1].to(device)
                xt_next = alpha_t_prev * x0_t + sigma_t_prev * randn_tensor(x0_t.shape, generator=generator, device=x0_t.device, dtype=x0_t.dtype)
                xs.append(xt_next.to("cpu"))

        t_prev_idx = torch.tensor(0, dtype=torch.int32)
        x_0 = self.denoiser.pred_x0(xt_next, t_prev_idx)
        nfe_count += 1
        decoded = self.denoiser.decode(x_0)
        return decoded.clamp(-1.0, 1.0), nfe_count

    def _get_schedule_jump(
        self,
        T_sampling,
        travel_length,
        travel_repeat
    ):
        
        jumps = {}
        for j in range(0, T_sampling - travel_length, travel_length):
            jumps[j] = travel_repeat - 1
            
        t = T_sampling
        ts = []
        while t >= 1:
            t = t - 1
            ts.append(t)

            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(travel_length):
                    t = t + 1
                    ts.append(t)

        ts.append(-1)
        self._check_times(ts, -1, T_sampling)

        return ts

    @staticmethod
    def _check_times(times, t_0, T_sampling):

        assert times[0] > times[1], (times[0], times[1])
        assert times[-1] == -1, times[-1]

        for t_last, t_cur in zip(times[:-1], times[1:]):
            assert abs(t_last - t_cur) == 1, (t_last, t_cur)

        for t in times:
            assert t >= t_0, (t, t_0)
            assert t <= T_sampling, (t, T_sampling)