from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

CropName = Literal["whole_page", "half_page", "tight_text"]

@dataclass(frozen = True)
class CropSpec:
    name: CropName
    width_frac: float | None = None
    height_frac: float | None = None
    padding_pt: float = 0.0

SHARED_CROPS: dict[CropName, CropSpec] = {
    "whole_page": CropSpec(name = "whole_page", width_frac = 1.0, height_frac = 1.0),
    "half_page": CropSpec(name = "half_page", width_frac = 1.0, height_frac = 0.5),
    "tight_text": CropSpec(name = "tight_text", width_frac = None, height_frac = None, padding_pt = 18.0),
}