from __future__ import annotations
import numpy as np
from src.crops import CropSpec

def apply_crop(image: np.ndarray, spec: CropSpec) -> tuple[np.ndarray, dict[str, object]]:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError()

    height_px, width_px = image.shape[:2]
    if spec.name == "whole_page":
        return image, {"crop_name": spec.name, "box_px": [0, 0, width_px, height_px]}

    raise NotImplementedError()