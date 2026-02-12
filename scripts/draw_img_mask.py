"""Interactively paint an inpainting mask over an image using the mouse."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="Path to the image to mask.")
    parser.add_argument("mask_path", type=Path, help="Path to save mask.")
    parser.add_argument("--brush-size", type=int, default=30, help="Brush radius in pixels (default: 20).")
    parser.add_argument("--invert", action="store_true", help="Start with fully masked image (paint to reveal).")
    return parser.parse_args()


def make_overlay(mask: np.ndarray) -> np.ndarray:
    """Return an RGBA overlay highlighting masked (zero) regions in red."""
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask < 0.5] = (1.0, 0.0, 0.0, 0.35)
    return overlay


def apply_brush(mask: np.ndarray, x: int, y: int, radius: int, erase: bool) -> None:
    """Modify the mask in place with a circular brush."""
    h, w = mask.shape
    y_grid, x_grid = np.ogrid[0:h, 0:w]
    dist_sq = (x_grid - x) ** 2 + (y_grid - y) ** 2
    region = dist_sq <= radius * radius
    mask[region] = 1.0 if erase else 0.0


def save_outputs(mask: np.ndarray, img_np: np.ndarray, mask_path: Path) -> Tuple[Path, Path]:
    mask_img = (mask * 255).astype(np.uint8)
    #masked = (img_np * mask[..., None]).astype(np.uint8)
    Image.fromarray(mask_img).save(mask_path)
    print(f"Saved mask to {mask_path}")


def main() -> None:
    args = parse_args()
    img = Image.open(args.image_path).convert("RGB")
    img_np = np.asarray(img, dtype=np.uint8)
    h, w = img_np.shape[:2]

    mask = np.zeros((h, w), dtype=np.float32) if args.invert else np.ones((h, w), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Left-drag to erase (set mask to 0). Right-drag to restore (mask=1). Press 's' to save, 'q' to quit.")
    ax.axis("off")
    base_img = ax.imshow(img_np)
    overlay_img = ax.imshow(make_overlay(mask))

    state = {"drawing": False, "erase": False}

    def on_press(event):
        if event.inaxes != ax:
            return
        if event.button not in {1, 3}:  # left or right
            return
        state["drawing"] = True
        state["erase"] = event.button == 3
        on_move(event)

    def on_release(event):
        state["drawing"] = False

    def on_move(event):
        if not state["drawing"] or event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        apply_brush(mask, x, y, max(1, args.brush_size), erase=state["erase"])
        overlay_img.set_data(make_overlay(mask))
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "s":
            save_outputs(mask, img_np, args.mask_path)
            
        elif event.key in {"q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)

    try:
        plt.show()
    finally:
        plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    main()
