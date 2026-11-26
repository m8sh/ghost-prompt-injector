from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from PIL import Image

DEFAULT_DEBUG_TARGET_SPAN = 160.0
DEFAULT_DEBUG_MAX_GAIN = 64.0

def save_composed_png(composed: np.ndarray, png_path: str | Path, dpi: int) -> Path:
    if composed.dtype != np.uint8:
        raise TypeError()
    if composed.ndim != 3 or composed.shape[2] != 3:
        raise ValueError()

    png_path = Path(png_path)
    png_path.parent.mkdir(parents = True, exist_ok = True)
    # Lossless PNG in plain RGB (no ICC profile) with exact DPI metadata.
    Image.fromarray(composed, mode = "RGB").save(png_path, format = "PNG", dpi = (dpi, dpi))
    return png_path

def build_debug_view(background: np.ndarray, composed: np.ndarray, mask: Image.Image,
                     target_span: float = DEFAULT_DEBUG_TARGET_SPAN,
                     max_gain: float = DEFAULT_DEBUG_MAX_GAIN) -> np.ndarray:
    if background.dtype != np.uint8 or composed.dtype != np.uint8:
        raise TypeError()
    if background.shape != composed.shape:
        raise ValueError()

    out = background.copy()
    mask_arr = np.asarray(mask, dtype = np.uint8)
    ys, xs = np.nonzero(mask_arr)
    if ys.size == 0:
        return out

    bg_px = background[ys, xs].astype(np.float32)
    diff = composed[ys, xs].astype(np.float32) - bg_px
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())
    if max_abs == 0.0:
        return out

    ref_level = float(np.percentile(abs_diff, 95))
    if ref_level == 0.0:
        ref_level = max_abs

    gain = max(1.0, min(max_gain, target_span / ref_level))
    out[ys, xs] = np.clip(np.rint(bg_px + diff * gain), 0, 255).astype(np.uint8)
    return out

def _srgb_to_linear(rgb_uint8: np.ndarray) -> np.ndarray:
    c = rgb_uint8.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))

def _srgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    lin = _srgb_to_linear(rgb_uint8 = rgb_uint8)
    x = (0.4124564 * lin[:, 0] + 0.3575761 * lin[:, 1] + 0.1804375 * lin[:, 2]) / 0.95047
    y = (0.2126729 * lin[:, 0] + 0.7151522 * lin[:, 1] + 0.0721750 * lin[:, 2]) / 1.00000
    z = (0.0193339 * lin[:, 0] + 0.1191920 * lin[:, 1] + 0.9503041 * lin[:, 2]) / 1.08883

    delta = 6.0 / 29.0
    threshold = delta ** 3
    slope = 1.0 / (3.0 * delta * delta)
    offset = 4.0 / 29.0

    fx = np.where(x > threshold, np.cbrt(x), slope * x + offset)
    fy = np.where(y > threshold, np.cbrt(y), slope * y + offset)
    fz = np.where(z > threshold, np.cbrt(z), slope * z + offset)

    l_star = 116.0 * fy - 16.0
    a_star = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)
    return np.stack((l_star, a_star, b_star), axis = -1)

def _relative_luminance(rgb_uint8: np.ndarray) -> np.ndarray:
    lin = _srgb_to_linear(rgb_uint8 = rgb_uint8)
    return 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]

def compute_contrast_metrics(background: np.ndarray, composed: np.ndarray, mask: Image.Image) -> dict[str, float | int]:
    if background.dtype != np.uint8 or composed.dtype != np.uint8:
        raise TypeError()
    if background.shape != composed.shape:
        raise ValueError()

    mask_arr = np.asarray(mask, dtype = np.uint8)
    ys, xs = np.nonzero(mask_arr)
    if ys.size == 0:
        return {
            "masked_pixel_count": 0,
            "core_pixel_count": 0,
            "max_channel_diff": 0,
            "core_mean_change": 0.0,
            "core_median_change": 0.0,
            "core_max_change": 0.0,
            "all_mean_change": 0.0,
            "contrast_ratio_mean": 1.0,
            "contrast_ratio_median": 1.0,
            "contrast_ratio_max": 1.0,
        }

    bg_all = background[ys, xs]
    comp_all = composed[ys, xs]
    lab_bg_all = _srgb_to_lab(rgb_uint8 = bg_all)
    lab_comp_all = _srgb_to_lab(rgb_uint8 = comp_all)
    delta_e_all = np.linalg.norm(lab_comp_all - lab_bg_all, axis = 1)

    core_selector = mask_arr[ys, xs] == 255
    core_count = int(core_selector.sum())
    if core_count > 0:
        bg_core = bg_all[core_selector]
        comp_core = comp_all[core_selector]
        delta_e_core = delta_e_all[core_selector]
    else:
        bg_core = bg_all
        comp_core = comp_all
        delta_e_core = delta_e_all

    luma_bg = _relative_luminance(rgb_uint8 = bg_core)
    luma_comp = _relative_luminance(rgb_uint8 = comp_core)
    ratios = (np.maximum(luma_bg, luma_comp) + 0.05) / (np.minimum(luma_bg, luma_comp) + 0.05)
    max_channel_diff = int(np.abs(comp_all.astype(np.int16) - bg_all.astype(np.int16)).max())

    return {
        "masked_pixel_count": int(ys.size),
        "core_pixel_count": core_count,
        "max_channel_diff": max_channel_diff,
        "core_mean_change": round(float(delta_e_core.mean()), 4),
        "core_median_change": round(float(np.median(delta_e_core)), 4),
        "core_max_change": round(float(delta_e_core.max()), 4),
        "all_mean_change": round(float(delta_e_all.mean()), 4),
        "contrast_ratio_mean": round(float(ratios.mean()), 4),
        "contrast_ratio_median": round(float(np.median(ratios)), 4),
        "contrast_ratio_max": round(float(ratios.max()), 4),
    }

def save_metrics_json(json_path: str | Path, payload: dict[str, object]) -> Path:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents = True, exist_ok = True)
    json_path.write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
    return json_path