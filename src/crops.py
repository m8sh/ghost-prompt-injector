from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np

CropName = Literal["whole_page", "half_page", "tight_text"]
CropPosition = Literal["top", "bottom", "left", "right", "center", "random"]

POINTS_PER_INCH = 72.0

@dataclass(frozen = True)
class CropSpec:
    name: CropName
    width_frac: float | None = None
    height_frac: float | None = None
    padding_pt: float = 0.0
    position: CropPosition = "top"

SHARED_CROPS: dict[CropName, CropSpec] = {
    "whole_page": CropSpec(name = "whole_page", width_frac = 1.0, height_frac = 1.0, position = "center"),
    "half_page": CropSpec(name = "half_page", width_frac = 1.0, height_frac = 0.5, position = "top"),
    "tight_text": CropSpec(name = "tight_text", width_frac = None, height_frac = None, padding_pt = 18.0,
                           position = "center"),
}

def compute_tight_text_box(content_map: np.ndarray, dpi: float,
                           padding_pt: float = SHARED_CROPS["tight_text"].padding_pt) -> (
                                                                                    tuple[int, int, int, int] | None):
    if content_map.ndim != 2:
        raise ValueError()
    ys, xs = np.nonzero(content_map)
    if ys.size == 0:
        return None
    height_px, width_px = content_map.shape
    pad_px = int(round(padding_pt * dpi / POINTS_PER_INCH))
    x0 = max(0, int(xs.min()) - pad_px)
    y0 = max(0, int(ys.min()) - pad_px)
    x1 = min(width_px, int(xs.max()) + 1 + pad_px)
    y1 = min(height_px, int(ys.max()) + 1 + pad_px)
    return (x0, y0, x1, y1)