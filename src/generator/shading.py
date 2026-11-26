from __future__ import annotations
from PIL import Image, ImageFilter
from dataclasses import dataclass
from typing import Literal

import numpy as np

ReferenceMode = Literal["pixel", "local_average"]
ColorSpace = Literal["rgb", "lab"]

DEFAULT_STRENGTH = 128.0
DEFAULT_LUMA_MIDPOINT = 128.0
DEFAULT_REFERENCE_MODE: ReferenceMode = "pixel"
DEFAULT_BLUR_RADIUS_PX = 6.0
DEFAULT_COLOR_SPACE: ColorSpace = "rgb"


@dataclass(frozen = True)
class ShadingConfig:
    strength: float = DEFAULT_STRENGTH
    reference_mode: ReferenceMode = DEFAULT_REFERENCE_MODE
    blur_radius_px: float = DEFAULT_BLUR_RADIUS_PX
    luma_midpoint: float = DEFAULT_LUMA_MIDPOINT
    color_space: ColorSpace = DEFAULT_COLOR_SPACE


def _validate_inputs(background: np.ndarray, config: ShadingConfig) -> None:
    if background.dtype != np.uint8:
        raise TypeError()
    if background.ndim != 3 or background.shape[2] != 3:
        raise ValueError()
    if config.strength < 0.0:
        raise ValueError()
    if config.blur_radius_px < 0.0:
        raise ValueError()
    if not (0.0 <= config.luma_midpoint <= 255.0):
        raise ValueError()
    if config.reference_mode not in ("pixel", "local_average"):
        raise ValueError()
    if config.color_space not in ("rgb", "lab"):
        raise ValueError()


def compute_reference_background(background: np.ndarray, config: ShadingConfig) -> np.ndarray:
    if config.reference_mode == "pixel" or config.blur_radius_px == 0.0:
        return background
    # Blur in 8-bit RGB via Pillow to avoid allocating a full-page float32 array.
    blurred = Image.fromarray(background, mode = "RGB").filter(ImageFilter.GaussianBlur(radius = config.blur_radius_px))
    return np.asarray(blurred, dtype = np.uint8)


def pick_shading_direction(reference_rgb: np.ndarray, luma_midpoint: float = DEFAULT_LUMA_MIDPOINT) -> np.ndarray:
    # Rec. 709 luminance on 0-255 scale: darken (-1) on light areas, lighten (+1) on dark areas.
    luma = 0.2126 * reference_rgb[:, 0] + 0.7152 * reference_rgb[:, 1] + 0.0722 * reference_rgb[:, 2]
    return np.where(luma >= luma_midpoint, -1.0, 1.0).astype(np.float32)


def _apply_color_space_offset(reference_rgb: np.ndarray, signed_offset: np.ndarray, color_space: ColorSpace) -> np.ndarray:
    if color_space == "rgb":
        return reference_rgb + signed_offset[:, np.newaxis]
    if color_space == "lab":
        # Future hook: convert (N, 3) sRGB to CIELAB, add signed_offset to L* (matching delta E), convert back to sRGB.
        raise NotImplementedError()
    raise ValueError()


def apply_shading(background: np.ndarray, mask: Image.Image, config: ShadingConfig | None = None) -> np.ndarray:
    if config is None:
        config = ShadingConfig()
    _validate_inputs(background = background, config = config)

    out = background.copy()
    if config.strength == 0.0:
        return out

    mask_arr = np.asarray(mask, dtype = np.uint8)
    ys, xs = np.nonzero(mask_arr)
    if ys.size == 0:
        return out

    ref_bg = compute_reference_background(background = background, config = config)
    bg_pixels = background[ys, xs].astype(np.float32)
    ref_pixels = ref_bg[ys, xs].astype(np.float32)

    directions = pick_shading_direction(reference_rgb = ref_pixels, luma_midpoint = config.luma_midpoint)
    signed_offset = directions * float(config.strength)
    target_rgb = np.clip(_apply_color_space_offset(reference_rgb = ref_pixels, signed_offset = signed_offset, color_space = config.color_space), 0.0, 255.0)

    weights = (mask_arr[ys, xs].astype(np.float32) / 255.0)[:, np.newaxis]
    blended = bg_pixels + weights * (target_rgb - bg_pixels)
    out[ys, xs] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return out


def build_overlay_rgba(background: np.ndarray, mask: Image.Image, config: ShadingConfig | None = None) -> np.ndarray:
    if config is None:
        config = ShadingConfig()
    _validate_inputs(background = background, config = config)

    mask_arr = np.asarray(mask, dtype = np.uint8)
    rgba = np.empty((background.shape[0], background.shape[1], 4), dtype = np.uint8)
    rgba[..., :3] = background
    rgba[..., 3] = 0

    ys, xs = np.nonzero(mask_arr)
    if ys.size == 0 or config.strength == 0.0:
        return rgba

    ref_bg = compute_reference_background(background = background, config = config)
    ref_pixels = ref_bg[ys, xs].astype(np.float32)

    directions = pick_shading_direction(reference_rgb = ref_pixels, luma_midpoint = config.luma_midpoint)
    signed_offset = directions * float(config.strength)
    target_rgb = _apply_color_space_offset(reference_rgb = ref_pixels, signed_offset = signed_offset, color_space = config.color_space)

    rgba[ys, xs, :3] = np.clip(np.rint(target_rgb), 0, 255).astype(np.uint8)
    rgba[ys, xs, 3] = mask_arr[ys, xs]
    return rgba