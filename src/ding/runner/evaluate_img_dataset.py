"""Hydra runner to evaluate image editing outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import torch
import hydra
import numpy as np
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from diffusers.utils import load_image
from torchmetrics.image.fid import FrechetInceptionDistance

from ding.utils import (
    abs_path,
    set_deterministic_seed,
    MetricsCalculator,
)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

# -----------------------------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------------------------

def _banner(message: str) -> str:
    line = "=" * len(message)
    return f"{line}\n{message}\n{line}"

def append_to_merged_summary(summary: Dict, out_dir: Path):
    merged_path = out_dir / "merged_summary.json"

    if merged_path.exists():
        data = json.loads(merged_path.read_text())
    else:
        data = []

    data.append(summary)
    merged_path.write_text(json.dumps(data, indent=2))

def load_image_np(path: Path) -> np.ndarray:
    """Load RGB image as uint8 [H, W, 3]."""
    img = load_image(str(path)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_mask_np(path: Path) -> np.ndarray:
    """
    Load PNG mask.

    Convention:
      0   = edit region
      255 = context
    """
    mask = Image.open(path).convert("L")
    mask = np.array(mask, dtype=np.float32) / 255.0
    return (mask > 0.5).astype(np.float32)


def to_uint8_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    """
    [H, W, 3] uint8 numpy → [1, 3, H, W] uint8 torch tensor on `device`
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return x.to(device=device, dtype=torch.uint8)

def extract_masked_nonblack_patches(
    img: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    min_edited_frac: float = 0.25,  # keep patch if >= 25% of pixels are edited
) -> List[np.ndarray]:
    """
    img: [H,W,3] uint8
    mask: [H,W] float {0,1} where 0=edited region, 1=context
    returns: list of patches that have enough edited pixels
    """
    # if you pass mask as [H,W,1], squeeze it
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    mask_edit = 1.0 - mask  # 1 in edited region, 0 in context
    H, W, C = img.shape
    ps = patch_size
    assert C == 3
    assert mask_edit.shape == (H, W)
    assert H % ps == 0 and W % ps == 0, "Image size must be divisible by patch_size"

    patches: List[np.ndarray] = []
    for y0 in range(0, H, ps):
        for x0 in range(0, W, ps):
            patch = img[y0:y0+ps, x0:x0+ps, :]
            m = mask_edit[y0:y0+ps, x0:x0+ps]  # float in {0,1}

            edited_frac = float(m.mean())
            if edited_frac < min_edited_frac:
                continue

            patches.append(patch)

    return patches

def extract_random_patches(
    gt_img: np.ndarray,
    gen_img: np.ndarray,
    patch_size: int = 256,
    n_patches: int = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract random aligned patches from GT and generated images.

    Args:
        gt_img, gen_img: uint8 arrays of shape [H, W, 3]
        patch_size: patch spatial size
        n_patches: number of random patches per image

    Returns:
        gt_patches, gen_patches: lists of uint8 patches [patch_size, patch_size, 3]
    """
    assert gt_img.shape == gen_img.shape
    H, W, C = gt_img.shape
    assert C == 3

    if H < patch_size or W < patch_size:
        return [], []

    gt_patches: List[np.ndarray] = []
    gen_patches: List[np.ndarray] = []

    for _ in range(n_patches):
        y0 = np.random.randint(0, H - patch_size + 1)
        x0 = np.random.randint(0, W - patch_size + 1)

        gt_patch = gt_img[y0:y0 + patch_size, x0:x0 + patch_size]
        gen_patch = gen_img[y0:y0 + patch_size, x0:x0 + patch_size]

        gt_patches.append(gt_patch)
        gen_patches.append(gen_patch)

    return gt_patches, gen_patches

# -----------------------------------------------------------------------------
# Core evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_image_edit(
    denoiser: str,
    sampler: str,
    steps: int,
    guidance_scale: float,
    gt_dir: Path,
    gen_dir: Path,
    metadata_path: Path,
    device: str,
    max_images: Optional[int],
    compute_fid: bool,
    compute_patch_fid: bool,
    compute_edit_patch_fid: bool,
    patch_size: int,
    n_patches: int,
    min_edited_frac: float,
) -> Dict:

    metrics = MetricsCalculator(device=device)
    metadata = json.loads(metadata_path.read_text())
    if max_images is not None:
        metadata = metadata[:max_images]

    # -------------------------------
    # FID metrics
    # -------------------------------
    if compute_fid:
        fid = FrechetInceptionDistance(feature=2048).to(device)
    
    if compute_patch_fid:
        patch_fid = FrechetInceptionDistance(feature=2048).to(device)

    if compute_edit_patch_fid:
        edit_patch_fid = FrechetInceptionDistance(feature=2048).to(device)

    totals: Dict[str, List[float]] = {
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
        "clip": [],
    }

    # -------------------------------
    # Loop over the data dir
    # -------------------------------
    count_edit_patches = 0
    count_patches = 0
    desc = f"Eval | {denoiser}+{sampler} | {steps} steps | {guidance_scale} gs"
    for item in tqdm(metadata, desc=desc, unit="img"):
        name = item["image"]
        mask_name = item["mask"]
        prompt = item["inpaint_prompt"]

        gt = load_image_np(gt_dir / name)           # [H, W, 3]
        gen = load_image_np(gen_dir / name)         # [H, W, 3]
        mask = load_mask_np(gt_dir / mask_name)     # [H, W]
        mask = mask[..., None]                      # [H, W, 1]
        assert gt.shape == gen.shape
        assert mask.shape[:2] == gt.shape[:2]

        # -------------------------------
        # Distortion metrics
        # -------------------------------
        totals["lpips"].append(metrics.calculate_lpips(gen, gt))
        totals["psnr"].append(metrics.calculate_psnr(gen, gt))
        totals["ssim"].append(metrics.calculate_ssim(gen, gt))
        totals["mse"].append(metrics.calculate_mse(gen, gt))
        totals["mae"].append(metrics.calculate_mae(gen, gt))
        
        totals["context_lpips"].append(metrics.calculate_lpips(gen, gt, mask))
        totals["context_psnr"].append(metrics.calculate_psnr(gen, gt, mask))
        totals["context_ssim"].append(metrics.calculate_ssim(gen, gt, mask))
        totals["context_mse"].append(metrics.calculate_mse(gen, gt, mask))
        totals["context_mae"].append(metrics.calculate_mae(gen, gt, mask))

        totals["clip"].append(metrics.calculate_clip_similarity(gen, prompt))

        # -------------------------------
        # FID and pFID updates
        # -------------------------------
        if compute_fid:
            fid.update(to_uint8_tensor(gt, device), real=True)
            fid.update(to_uint8_tensor(gen, device), real=False)

        # patch fid on random patches 
        if compute_patch_fid:
            gt_patches, gen_patches = extract_random_patches(gt, gen, patch_size, n_patches)
            if len(gt_patches) == 0:
                raise ValueError
            assert len(gt_patches) == len(gen_patches)
            count_patches += len(gt_patches)
            for p_gt, p_gen in zip(gt_patches, gen_patches):
                patch_fid.update(to_uint8_tensor(p_gt, device), real=True)
                patch_fid.update(to_uint8_tensor(p_gen, device), real=False)

        # patch fid on patches from edited regions 
        if compute_edit_patch_fid:
            gt_edit_patches = extract_masked_nonblack_patches(gt, mask, patch_size, min_edited_frac)
            gen_edit_patches = extract_masked_nonblack_patches(gen, mask, patch_size, min_edited_frac)
            if len(gt_edit_patches) == 0: # ignore the pfid compute image if no patch has been selected
                continue
            assert len(gt_edit_patches) == len(gen_edit_patches)
            count_edit_patches += len(gt_edit_patches)
            for p_gt, p_gen in zip(gt_edit_patches, gen_edit_patches):
                edit_patch_fid.update(to_uint8_tensor(p_gt, device), real=True)
                edit_patch_fid.update(to_uint8_tensor(p_gen, device), real=False)
        
    mean = {k: round(float(np.mean(v)), 3) for k, v in totals.items()}
    std = {k: round(float(np.std(v)), 3) for k, v in totals.items()}

    summary_mean = mean.copy()
    summary_std = std.copy()

    if compute_fid:
        summary_mean["fid"] = round(float(fid.compute()), 3)
        summary_std["fid"] = 0.0

    if compute_patch_fid:
        summary_mean["patch_fid"] = round(float(patch_fid.compute()), 3)
        summary_std["patch_fid"] = 0.0
        print(f"patch_fid was calculated over {count_patches} patches")

    if compute_edit_patch_fid:
        summary_mean["edit_patch_fid"] = round(float(edit_patch_fid.compute()), 3)
        summary_std["edit_patch_fid"] = 0.0
        print(f"edit_patch_fid was calculated over {count_edit_patches} patches")

    final_summary = {
        "denoiser": denoiser,
        "sampler": sampler,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "mean": summary_mean,
        "std": summary_std,
    }

    return {"per_image": totals, "final_summary": final_summary,}


# -----------------------------------------------------------------------------
# Hydra entry
# -----------------------------------------------------------------------------

@hydra.main(config_path=str(CONFIG_DIR), config_name="evaluate_img_dataset", version_base=None)
def main(cfg: DictConfig) -> None:
    print(_banner("ding.runner.evaluate_img_dataset"))
    print(OmegaConf.to_yaml(cfg, resolve=True))
    set_deterministic_seed(int(cfg.get("seed", 0)))

    results = evaluate_image_edit(
        denoiser=cfg.experiment.denoiser,
        sampler=cfg.experiment.sampler,
        steps=cfg.experiment.steps,
        guidance_scale=cfg.experiment.guidance_scale,
        gt_dir=abs_path(cfg.data.gt_dir),
        gen_dir=abs_path(cfg.data.gen_dir),
        metadata_path=abs_path(cfg.data.metadata_path),
        max_images=cfg.data.max_images,
        device=cfg.device,
        compute_fid=cfg.metrics.compute_fid,
        compute_patch_fid=cfg.metrics.compute_patch_fid,
        compute_edit_patch_fid=cfg.metrics.compute_edit_patch_fid,
        patch_size=cfg.metrics.pfid_patch_size,
        n_patches=cfg.metrics.n_random_patches,
        min_edited_frac=cfg.metrics.pfid_min_edited_frac,
    )

    out = abs_path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results["per_image"], indent=2))
    (out / "summary.json").write_text(json.dumps(results["final_summary"], indent=2))

    append_to_merged_summary(results["final_summary"], out.parent)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
