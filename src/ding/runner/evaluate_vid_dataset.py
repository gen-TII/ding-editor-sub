"""Hydra runner to evaluate video editing outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
from PIL import Image
from diffusers.utils import load_video

from ding.utils import (
    abs_path,
    set_deterministic_seed,
    MetricsCalculator,
    calculate_i3d_activations,
    calculate_vfid,
    init_i3d_model,
)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

# -----------------------------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------------------------

def _banner(message: str) -> str:
    line = "=" * len(message)
    return f"{line}\n{message}\n{line}"

def load_video_to_numpy(path):
    video = load_video(path)  
    frames = [np.array(frame.convert("RGB")) for frame in video]
    return np.stack(frames) # [T, H, W, 3]

def append_to_merged_summary(summary: Dict, out_dir: Path):
    merged_path = out_dir / "merged_summary.json"

    if merged_path.exists():
        data = json.loads(merged_path.read_text())
    else:
        data = []

    data.append(summary)
    merged_path.write_text(json.dumps(data, indent=2))

# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_video_edit(
    denoiser: str,
    sampler: str,
    steps: int,
    guidance_scale: float,
    gt_dir: Path,
    metadata_path: Path,
    gen_dir: Path,
    max_videos: Optional[int],
    device: str,
    compute_vfid: bool,
    vfid_ckpt: str,
) -> Dict:

    metadata = json.loads(metadata_path.read_text())
    if max_videos is not None:
        metadata = metadata[:max_videos]

    metrics = MetricsCalculator(device)
    if compute_vfid:
        i3d_model = init_i3d_model(vfid_ckpt).to(device)

    totals = {
        "lpips": [],
        "psnr": [],
        "ssim": [],
        "mse": [],
        "mae": [],
        "context_lpips": [],
        "context_psnr": [],
        "context_ssim": [],
        "context_mse": [],
        "context_mae": [],
        "temporal_consistency": [],
        "clip": [],
    }

    real_i3d, fake_i3d = [], []
    desc = f"Eval | {denoiser}+{sampler} | {steps} steps | {guidance_scale} gs"
    for item in tqdm(metadata, desc=desc, unit="vid"):
        prompt = item["inpaint_prompt"]
        vid = item["video"]
        gt_path =  gt_dir / vid
        mask_path = gt_dir / item["mask"] # 0 = edit, 1 = context
        gen_path = gen_dir / vid

        # load videos and mask and asumes they are aligned (length and resolution) with edited videos
        gt = load_video_to_numpy(str(gt_path))      # [T, H, W, 3]
        gen = load_video_to_numpy(str(gen_path))    # [T, H, W, 3]
        mask = np.load(mask_path)                   # [T, H, W]
        mask = mask[..., None]                      # [T, H, W, 1]
        assert gt.shape == gen.shape
        assert mask.shape[:3] == gt.shape[:3]

        per_video = {k: [] for k in totals if "temporal" not in k}
        for i in range(gen.shape[0]):
            per_video["lpips"].append(metrics.calculate_lpips(gen[i], gt[i]))
            per_video["psnr"].append(metrics.calculate_psnr(gen[i], gt[i]))
            per_video["ssim"].append(metrics.calculate_ssim(gen[i], gt[i]))
            per_video["mse"].append(metrics.calculate_mse(gen[i], gt[i]))
            per_video["mae"].append(metrics.calculate_mae(gen[i], gt[i]))
            
            per_video["context_lpips"].append(metrics.calculate_lpips(gen[i], gt[i], mask[i]))
            per_video["context_psnr"].append(metrics.calculate_psnr(gen[i], gt[i], mask[i]))
            per_video["context_ssim"].append(metrics.calculate_ssim(gen[i], gt[i], mask[i]))
            per_video["context_mse"].append(metrics.calculate_mse(gen[i], gt[i], mask[i]))
            per_video["context_mae"].append(metrics.calculate_mae(gen[i], gt[i], mask[i]))

            per_video["clip"].append(metrics.calculate_clip_similarity(gen[i], prompt))

        for k in per_video:
            totals[k].append(float(np.mean(per_video[k])))

        totals["temporal_consistency"].append(
            metrics.calculate_temporal_consistency(gen)
        )

        if compute_vfid:
            gt_pil = [Image.fromarray(f) for f in gt]
            gen_pil = [Image.fromarray(f) for f in gen]
            r, f = calculate_i3d_activations(
                gt_pil, gen_pil, i3d_model, device
            )
            real_i3d.append(r)
            fake_i3d.append(f)

    mean = {k: round(float(np.mean(v)), 3) for k, v in totals.items()}
    std = {k: round(float(np.std(v)), 3) for k, v in totals.items()}

    summary_mean = mean.copy()
    summary_std = std.copy()

    if compute_vfid:
        summary_mean["vfid"] = round(float(calculate_vfid(real_i3d, fake_i3d)), 3)
        summary_std["fid"] = 0.0

    final_summary = {
        "denoiser": denoiser,
        "sampler": sampler,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "mean": summary_mean,
        "std": summary_std,
    }
    
    return {"per_video": totals, "final_summary": final_summary}


@hydra.main(config_path=str(CONFIG_DIR), config_name="evaluate_vid_dataset", version_base=None)
def main(cfg: DictConfig) -> None:
    print(_banner("ding.runner.evaluate_vid_dataset"))
    print(OmegaConf.to_yaml(cfg, resolve=True))
    set_deterministic_seed(int(cfg.get("seed", 0)))

    results = evaluate_video_edit(
        denoiser=cfg.experiment.denoiser,
        sampler=cfg.experiment.sampler,
        steps=cfg.experiment.steps,
        guidance_scale=cfg.experiment.guidance_scale,
        gt_dir=abs_path(cfg.data.gt_dir),
        metadata_path=abs_path(cfg.data.metadata_path),
        gen_dir=abs_path(cfg.data.gen_dir),
        max_videos=cfg.data.max_videos,
        device=cfg.device,
        compute_vfid=cfg.metrics.compute_vfid,
        vfid_ckpt=str(cfg.metrics.vfid_ckpt),
    )

    out = abs_path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results["per_video"], indent=2))
    (out / "summary.json").write_text(json.dumps(results["final_summary"], indent=2))

    append_to_merged_summary(results["final_summary"], out.parent)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()