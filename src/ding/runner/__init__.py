"""Entry points for command-line runners."""

from __future__ import annotations

from typing import Any


def inpaint_img_main(*args: Any, **kwargs: Any):
    from .inpaint_img import main

    return main(*args, **kwargs)

def inpaint_img_dataset_main(*args: Any, **kwargs: Any):
    from .inpaint_img_dataset import main

    return main(*args, **kwargs)

def evaluate_img_dataset_main(*args: Any, **kwargs: Any):
    from .evaluate_img_dataset import main

    return main(*args, **kwargs)

def inpaint_vid_main(*args: Any, **kwargs: Any):
    from .inpaint_vid import main

    return main(*args, **kwargs)

def inpaint_vid_dataset_main(*args: Any, **kwargs: Any):
    from .inpaint_vid_dataset import main

    return main(*args, **kwargs)

def evaluate_vid_dataset_main(*args: Any, **kwargs: Any):
    from .evaluate_vid_dataset import main

    return main(*args, **kwargs)

def inpaint_audio_main(*args: Any, **kwargs: Any):
    from .inpaint_audio import main

    return main(*args, **kwargs)

__all__ = [
    "inpaint_img_main",
    "inpaint_img_dataset_main",
    "evaluate_img_dataset_main",
    "inpaint_vid_main",
    "inpaint_vid_dataset_main",
    "evaluate_vid_dataset_main",
    "inpaint_audio_main",
]
