"""Stable Audio 1 audio denoiser wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor
import numpy as np

from diffusers.utils.torch_utils import randn_tensor
from .base import BaseDenoiser
from diffusers import StableAudioPipeline


class StableAudio1Denoiser(BaseDenoiser):
    """Wrapper around the SAO audio generation pipeline."""

    default_model_id: str = "stabilityai/stable-audio-open-1.0"

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
        pipeline = StableAudioPipeline.from_pretrained(model_id, **load_kwargs)

        pipeline, device_obj = self._setup_pipeline_optimizations(
            pipeline=pipeline,
            device=device,
            enable_model_cpu_offload=model_cpu_offload,
            enable_sequential_cpu_offload=sequential_cpu_offload,
            enable_attention_slicing=attention_slicing,
            enable_vae_slicing=vae_slicing,
        )
        
        text_encoder = pipeline.text_encoder
        projection_model = pipeline.projection_model
        transformer = pipeline.transformer
        vae = pipeline.vae
        scheduler = pipeline.scheduler

        audio_channels = vae.config.audio_channels
        num_channels_vae = transformer.config.in_channels
        sample_rate = vae.config.sampling_rate
        downsample_ratio = vae.hop_length
        waveform_length = int(transformer.config.sample_size)
        audio_vae_length = waveform_length * downsample_ratio
        max_audio_length_in_s = audio_vae_length / sample_rate
        rotary_embed_dim = transformer.config.attention_head_dim // 2
        
        timesteps_desc = scheduler.timesteps.to(device=device_obj, dtype=torch.float32)
        sigmas_desc = scheduler.sigmas.to(device=device_obj, dtype=torch.float64)
        net_timesteps = timesteps_desc.flip(0)
        sigmas_f64 = sigmas_desc.flip(0)
        alphas_f64 = torch.ones_like(sigmas_f64)
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
        self.projection_model = projection_model
        self.scheduler = scheduler
        self.transformer = transformer
        self.vae = vae
    
        self.device = device_obj
        self.dtype = dtype

        self.audio_channels = audio_channels
        self.num_channels_vae = num_channels_vae
        self.sample_rate = sample_rate
        self.downsample_ratio = downsample_ratio
        self.waveform_length = waveform_length
        self.audio_vae_length = audio_vae_length
        self.max_audio_length_in_s = max_audio_length_in_s
        self.rotary_embed_dim = rotary_embed_dim

        self.alphas_f64 = alphas_f64
        self.sigmas_f64 = sigmas_f64
        self.net_timesteps = net_timesteps
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = timesteps
        self._prepare_networks()

    # ------------------------------------------------------------------
    # Pipeline memory optim and setup for inference
    # ------------------------------------------------------------------

    def _setup_pipeline_optimizations(
        self,
        pipeline: StableAudioPipeline,
        device: str | torch.device,
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False,
        enable_attention_slicing: bool = False,
        enable_vae_slicing: bool = False,
    ) -> tuple[StableAudioPipeline, torch.device]:
        
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
            self.projection_model,
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
        self.has_negative_prompts = negative_prompts is not None
        self.batch_size = len(prompts)

        self.prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompts,
            negative_prompt=negative_prompts,
            do_classifier_free_guidance=self.do_true_cfg,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, audios: Tensor, generator: torch.Generator) -> Tensor:
        input_dtype = audios.dtype
        audios = audios.to(self.dtype)
        latents = self.vae.encode(audios, return_dict=False)[0]
        latents = latents.sample(generator=generator)
        return latents.to(input_dtype).to(self.device)
    
    def decode(self, latents: Tensor) -> Tensor:
        input_dtype = latents.dtype
        latents = latents.to(self.dtype)
        audios = self.vae.decode(latents, return_dict=False)[0]
        audios = audios[:, :, self.waveform_start:self.waveform_end]
        return audios.to(input_dtype).to(self.device)

    # ------------------------------------------------------------------
    # Predict velocity
    # ------------------------------------------------------------------

    def _resolve_step(self, t: Tensor | int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Returns `(step_index, timestep_value, sigma_value)` for this denoiser.

        - `step_index` indexes into `self.net_timesteps`/`self.sigmas` (inference-step index).
        - `timestep_value` is the float timestep embedding value expected by SAO transformer (scheduler.timesteps).
        - `sigma_value` is the noise level (scheduler.sigmas) for EDM-style preconditioning.
        """

        if isinstance(t, Tensor):
            t_flat = t.reshape(-1)
            if t_flat.numel() != 1:
                raise ValueError("StableAudio1Denoiser only supports a single timestep per call.")
            if t_flat.dtype.is_floating_point:
                t_val = t_flat[0].to(device=self.device, dtype=torch.float32)
                timesteps = self.net_timesteps.to(device=t_val.device, dtype=torch.float32)
                step_idx = int((timesteps - t_val).abs().argmin().item())
            else:
                step_idx = int(t_flat[0].item())
        else:
            step_idx = int(t)

        timestep_value = self.net_timesteps[step_idx].to(device=self.device, dtype=torch.float32)
        sigma_value = self.sigmas_f64[step_idx].to(device=self.device, dtype=torch.float32)
        return step_idx, timestep_value, sigma_value

    def _predict_model_output(self, latents: Tensor, t: Tensor | int) -> tuple[Tensor, torch.Tensor]:
        """
        Runs the SAO transformer and returns `(model_output, sigma_value)`.

        `model_output` is in the scheduler's configured `prediction_type` (commonly `v_prediction` for SAO).
        """

        input_dtype = latents.dtype
        latents = latents.to(device=self.device, dtype=self.prompt_embeds.dtype)

        _, timestep_value, sigma_value = self._resolve_step(t)

        if self.do_true_cfg:
            latents = torch.cat([latents] * 2, dim=0)

        # Match `StableAudioPipeline`: precondition inputs using EDM-style scaling and feed the scheduler's timestep.
        latents_in = self.scheduler.precondition_inputs(latents, sigma_value)
        timestep = timestep_value.reshape(1)  # pipeline passes a single scalar `t`

        model_output = self.transformer(
            hidden_states=latents_in,
            timestep=timestep,
            encoder_hidden_states=self.text_audio_duration_embeds,
            global_hidden_states=self.audio_duration_embeds,
            rotary_embedding=self.rotary_embedding,
            return_dict=False,
        )[0]

        if self.do_true_cfg:
            model_output_uncond, model_output_cond = model_output.chunk(2)
            model_output = model_output_uncond + self.guidance_scale * (model_output_cond - model_output_uncond)

        return model_output.to(dtype=input_dtype), sigma_value

    def pred_x0(self, latents: Tensor, t: Tensor | int) -> Tensor:
        """
        Stable Audio Open uses an EDM-style sigma parameterization (VE-like): x = x0 + sigma * eps.

        The transformer is commonly trained with `prediction_type="v_prediction"`, so the correct x0 estimate is the
        scheduler's `precondition_outputs(sample, model_output, sigma)`.
        """

        input_dtype = latents.dtype
        sample = latents.to(device=self.device, dtype=self.prompt_embeds.dtype)
        model_output, sigma_value = self._predict_model_output(latents, t)
        model_output = model_output.to(device=self.device, dtype=self.prompt_embeds.dtype)

        x0 = self.scheduler.precondition_outputs(sample, model_output, sigma_value)
        return x0.to(dtype=input_dtype)

    def pred_velocity(self, latents: Tensor, t: Tensor | int) -> Tensor:
        """
        Returns the *raw* transformer output, i.e. whatever the scheduler was configured to predict.

        For Stable Audio Open this is commonly `prediction_type="v_prediction"`, so the returned tensor is the EDM
        "v-prediction" parameterization (not rectified-flow velocity).

        Use `pred_x0()` to get the denoised estimate and `pred_x1()` to get the unit-variance noise term `eps`.
        """

        model_output, _ = self._predict_model_output(latents, t)
        return model_output

    def pred_x1(self, latents: Tensor, t: Tensor | int) -> Tensor:
        """
        Returns the unit-variance noise term `eps` for the VE forward process:

            x = x0 + sigma * eps  =>  eps = (x - x0) / sigma
        """

        input_dtype = latents.dtype
        sample = latents.to(device=self.device, dtype=self.prompt_embeds.dtype)
        model_output, sigma_value = self._predict_model_output(latents, t)
        model_output = model_output.to(device=self.device, dtype=self.prompt_embeds.dtype)

        x0 = self.scheduler.precondition_outputs(sample, model_output, sigma_value)
        eps = (sample - x0) / sigma_value.clamp_min(1e-8)
        return eps.to(dtype=input_dtype)
    

    # ------------------------------------------------------------------
    # Setup input params and timesteps
    # ------------------------------------------------------------------
    
    def set_audio_params(
            self,
            audio_start_in_s: float = 0.0,
            audio_end_in_s: float = 10.0,
            n_wavs_per_prompt: int = 1
        ) -> None:

        if audio_end_in_s is None:
            audio_end_in_s = self.max_audio_length_in_s
        if audio_end_in_s - audio_start_in_s > self.max_audio_length_in_s:
            raise ValueError(
                f"The total audio length requested ({audio_end_in_s - audio_start_in_s}s) is longer than the model maximum possible length ({self.max_audio_length_in_s}). Make sure that 'audio_end_in_s-audio_start_in_s<={self.max_audio_length_in_s}'."
            )
        self.waveform_start = int(audio_start_in_s * self.sample_rate)
        self.waveform_end = int(audio_end_in_s * self.sample_rate)    

        # Encode duration
        seconds_start_hidden_states, seconds_end_hidden_states = self.pipeline.encode_duration(
            audio_start_in_s,
            audio_end_in_s,
            self.device,
            self.do_true_cfg and self.has_negative_prompts,
            self.batch_size,
        )
        # Create text_audio_duration_embeds and audio_duration_embeds
        text_audio_duration_embeds = torch.cat(
            [self.prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
        )
        audio_duration_embeds = torch.cat([seconds_start_hidden_states, seconds_end_hidden_states], dim=2)

        # In case of classifier free guidance without negative prompt, we need to create unconditional embeddings and
        # to concatenate it to the embeddings
        if self.do_true_cfg and not self.has_negative_prompts:
            negative_text_audio_duration_embeds = torch.zeros_like(
                text_audio_duration_embeds, device=text_audio_duration_embeds.device
            )
            text_audio_duration_embeds = torch.cat(
                [negative_text_audio_duration_embeds, text_audio_duration_embeds], dim=0
            )
            audio_duration_embeds = torch.cat([audio_duration_embeds, audio_duration_embeds], dim=0)

        bs_embed, seq_len, hidden_size = text_audio_duration_embeds.shape
        # duplicate audio_duration_embeds and text_audio_duration_embeds for each generation per prompt, using mps friendly method
        text_audio_duration_embeds = text_audio_duration_embeds.repeat(1, n_wavs_per_prompt, 1)
        text_audio_duration_embeds = text_audio_duration_embeds.view(
            bs_embed * n_wavs_per_prompt, seq_len, hidden_size
        )
        audio_duration_embeds = audio_duration_embeds.repeat(1, n_wavs_per_prompt, 1)
        audio_duration_embeds = audio_duration_embeds.view(
            bs_embed * n_wavs_per_prompt, -1, audio_duration_embeds.shape[-1]
        )

        rotary_seq_len = self.waveform_length + audio_duration_embeds.shape[1]
        self.rotary_embedding = self.get_1d_rotary_pos_embed(
            self.rotary_embed_dim,
            rotary_seq_len,
            use_real=True,
            repeat_interleave_real=False,
        )
        self.audio_duration_embeds = audio_duration_embeds
        self.text_audio_duration_embeds = text_audio_duration_embeds

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
        self.alphas_f64 = torch.ones_like(self.sigmas_f64)
        self.alphas = self.alphas_f64.to(self.device, self.dtype)
        self.sigmas = self.sigmas_f64.to(self.device, self.dtype)
        self.timesteps = torch.arange(n_steps, dtype=torch.int64, device=self.device)
        
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    
    @staticmethod
    def get_1d_rotary_pos_embed(
        dim: int,
        pos: Union[np.ndarray, int],
        theta: float = 10000.0,
        use_real=False,
        linear_factor=1.0,
        ntk_factor=1.0,
        repeat_interleave_real=True,
        freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

        This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
        index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
        data type.

        Args:
            dim (`int`): Dimension of the frequency tensor.
            pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
            theta (`float`, *optional*, defaults to 10000.0):
                Scaling factor for frequency computation. Defaults to 10000.0.
            use_real (`bool`, *optional*):
                If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            linear_factor (`float`, *optional*, defaults to 1.0):
                Scaling factor for the context extrapolation. Defaults to 1.0.
            ntk_factor (`float`, *optional*, defaults to 1.0):
                Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
            repeat_interleave_real (`bool`, *optional*, defaults to `True`):
                If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
                Otherwise, they are concateanted with themselves.
            freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
                the dtype of the frequency tensor.
        Returns:
            `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
        """
        assert dim % 2 == 0

        if isinstance(pos, int):
            pos = torch.arange(pos)
        if isinstance(pos, np.ndarray):
            pos = torch.from_numpy(pos)  # type: ignore  # [S]

        theta = theta * ntk_factor
        freqs = (
            1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
        )  # [D/2]
        freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
        is_npu = freqs.device.type == "npu"
        if is_npu:
            freqs = freqs.float()
        if use_real and repeat_interleave_real:
            # flux, hunyuan-dit, cogvideox
            freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
            freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
            return freqs_cos, freqs_sin
        elif use_real:
            # stable audio, allegro
            freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
            freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
            return freqs_cos, freqs_sin
        else:
            # lumina
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
            return freqs_cis

    # ------------------------------------------------------------------
    # Sampling pipelines
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_manual_wrapper(
        self,
        generator: torch.Generator,
        prompt: str,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 200,
        audio_end_in_s: Optional[float] = None,
        audio_start_in_s: Optional[float] = 0.0,
    ) -> Tensor:
        """Manual Stable Audio Open loop that reuses this wrapper's prompt/text/latents helpers."""

        # OK - Initial setup
        self.pipeline.set_progress_bar_config(disable=False)
        batch_size = 1
        n_wavs_per_prompt = 1

        # OK - Set prompt & audio params
        self.set_prompt(
            prompts=prompt,
            negative_prompts=negative_prompt,
            guidance_scale=guidance_scale,
        )

        self.set_audio_params(
            audio_start_in_s=audio_start_in_s,
            audio_end_in_s = audio_end_in_s,
            n_wavs_per_prompt=n_wavs_per_prompt,
        )

        # OK - Prepare timesteps
        self.set_timesteps(num_inference_steps)
        timesteps = self.net_timesteps.flip(0)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # OK - Init latents
        latent_shape = (batch_size*n_wavs_per_prompt, self.num_channels_vae, self.waveform_length)
        latents = randn_tensor(latent_shape, generator=generator, device=self.device, dtype=self.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        # OK - Denosing loop
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(
            generator=generator,
            eta=0.0
        )
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = self.pred_velocity(latents, t)
                latents = self.scheduler.step(
                    noise_pred,
                    t, latents,
                    **extra_step_kwargs,
                ).prev_sample
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()       

        # OK - Decode latents
        audio = self.decode(latents)
        if isinstance(audio, list):
            audio = audio[0]
        return audio.to(self.device, self.pipeline.dtype)
    
    @torch.no_grad()
    def generate_pipeline_baseline(
        self,
        generator: torch.Generator,
        prompt: str,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 200,
        audio_end_in_s: Optional[float] = None,
        audio_start_in_s: Optional[float] = 0.0,
    ) -> Tensor:
        """Run the official diffusers Stable Audio Open pipeline loop and return the decoded audio tensor."""

        self.pipeline.set_progress_bar_config(disable=False)
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            audio_end_in_s=audio_end_in_s,
            audio_start_in_s=audio_start_in_s,
            output_type="pt",
            return_dict=True,
            generator=generator,
        )
        audio = result.audios
        if isinstance(audio, list):
            audio = audio[0]
        return audio.to(self.device, self.dtype)
