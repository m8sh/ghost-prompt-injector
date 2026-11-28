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
    c = rgb_uint8.astype(np.float64) / 255.0
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

def _ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    l1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    l2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar ** 7
    pow25_7 = 25.0 ** 7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + pow25_7)))

    a1_p = (1.0 + g) * a1
    a2_p = (1.0 + g) * a2
    c1_p = np.hypot(a1_p, b1)
    c2_p = np.hypot(a2_p, b2)

    h1_p = np.degrees(np.arctan2(b1, a1_p)) % 360.0
    h2_p = np.degrees(np.arctan2(b2, a2_p)) % 360.0

    dl_p = l2 - l1
    dc_p = c2_p - c1_p

    c_prod = c1_p * c2_p
    dh_diff = h2_p - h1_p
    dh_p = np.where(c_prod == 0.0, 0.0,
                    np.where(np.abs(dh_diff) <= 180.0, dh_diff,
                             np.where(dh_diff > 180.0, dh_diff - 360.0, dh_diff + 360.0)))
    d_big_h_p = 2.0 * np.sqrt(c_prod) * np.sin(np.radians(0.5 * dh_p))

    l_bar_p = 0.5 * (l1 + l2)
    c_bar_p = 0.5 * (c1_p + c2_p)

    h_sum = h1_p + h2_p
    h_bar_p = np.where(c_prod == 0.0, h_sum,
                       np.where(np.abs(h1_p - h2_p) <= 180.0, 0.5 * h_sum,
                                np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0))))

    t = (1.0
         - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
         + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0)))

    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    c_bar_p7 = c_bar_p ** 7
    r_c = 2.0 * np.sqrt(c_bar_p7 / (c_bar_p7 + pow25_7))

    l_offset = l_bar_p - 50.0
    s_l = 1.0 + (0.015 * l_offset * l_offset) / np.sqrt(20.0 + l_offset * l_offset)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c

    term_l = dl_p / s_l
    term_c = dc_p / s_c
    term_h = d_big_h_p / s_h
    return np.sqrt(np.maximum(0.0, term_l * term_l + term_c * term_c + term_h * term_h + r_t * term_c * term_h))

def _relative_luminance(rgb_uint8: np.ndarray) -> np.ndarray:
    lin = _srgb_to_linear(rgb_uint8 = rgb_uint8)
    return 0.2126 * lin[:, 0] + 0.7152 * lin[:, 1] + 0.0722 * lin[:, 2]

def _empty_metrics() -> dict[str, float | int | str]:
    return {
        "delta_e_formula": "CIEDE2000",
        "masked_pixel_count": 0,
        "core_pixel_count": 0,
        "max_channel_diff": 0,
        "ciede2000_core_mean": 0.0,
        "ciede2000_core_median": 0.0,
        "ciede2000_core_max": 0.0,
        "ciede2000_all_mean": 0.0,
        "cie76_core_mean": 0.0,
        "cie76_core_median": 0.0,
        "cie76_core_max": 0.0,
        "cie76_all_mean": 0.0,
        "contrast_ratio_mean": 1.0,
        "contrast_ratio_median": 1.0,
        "contrast_ratio_max": 1.0,
    }

def _compute_subset_metrics(bg_all: np.ndarray, comp_all: np.ndarray,
                            mask_vals: np.ndarray) -> dict[str, float | int | str]:
    if bg_all.shape[0] == 0:
        return _empty_metrics()

    lab_bg_all = _srgb_to_lab(rgb_uint8 = bg_all)
    lab_comp_all = _srgb_to_lab(rgb_uint8 = comp_all)
    de2000_all = _ciede2000(lab1 = lab_bg_all, lab2 = lab_comp_all)
    de76_all = np.linalg.norm(lab_comp_all - lab_bg_all, axis = 1)

    core_selector = mask_vals == 255
    core_count = int(core_selector.sum())
    if core_count > 0:
        bg_core = bg_all[core_selector]
        comp_core = comp_all[core_selector]
        de2000_core = de2000_all[core_selector]
        de76_core = de76_all[core_selector]
    else:
        bg_core = bg_all
        comp_core = comp_all
        de2000_core = de2000_all
        de76_core = de76_all

    luma_bg = _relative_luminance(rgb_uint8 = bg_core)
    luma_comp = _relative_luminance(rgb_uint8 = comp_core)
    ratios = (np.maximum(luma_bg, luma_comp) + 0.05) / (np.minimum(luma_bg, luma_comp) + 0.05)
    max_channel_diff = int(np.abs(comp_all.astype(np.int16) - bg_all.astype(np.int16)).max())

    return {
        "delta_e_formula": "CIEDE2000",
        "masked_pixel_count": int(bg_all.shape[0]),
        "core_pixel_count": core_count,
        "max_channel_diff": max_channel_diff,
        "ciede2000_core_mean": round(float(de2000_core.mean()), 4),
        "ciede2000_core_median": round(float(np.median(de2000_core)), 4),
        "ciede2000_core_max": round(float(de2000_core.max()), 4),
        "ciede2000_all_mean": round(float(de2000_all.mean()), 4),
        "cie76_core_mean": round(float(de76_core.mean()), 4),
        "cie76_core_median": round(float(np.median(de76_core)), 4),
        "cie76_core_max": round(float(de76_core.max()), 4),
        "cie76_all_mean": round(float(de76_all.mean()), 4),
        "contrast_ratio_mean": round(float(ratios.mean()), 4),
        "contrast_ratio_median": round(float(np.median(ratios)), 4),
        "contrast_ratio_max": round(float(ratios.max()), 4),
    }

def compute_contrast_metrics(background: np.ndarray, composed: np.ndarray, mask: Image.Image,
                             content_map: np.ndarray | None = None,
                             split_by_content: bool = False) -> dict[str, object]:
    if background.dtype != np.uint8 or composed.dtype != np.uint8:
        raise TypeError()
    if background.shape != composed.shape:
        raise ValueError()

    mask_arr = np.asarray(mask, dtype = np.uint8)
    ys, xs = np.nonzero(mask_arr)
    bg_all = background[ys, xs]
    comp_all = composed[ys, xs]
    mask_vals = mask_arr[ys, xs]

    result: dict[str, object] = _compute_subset_metrics(bg_all = bg_all, comp_all = comp_all, mask_vals = mask_vals)
    if split_by_content and content_map is not None:
        content_bool = np.asarray(content_map, dtype = bool)
        if content_bool.shape != mask_arr.shape:
            raise ValueError()
        over_text = content_bool[ys, xs]
        on_paper = ~over_text
        result["prompt_on_paper"] = _compute_subset_metrics(bg_all = bg_all[on_paper], comp_all = comp_all[on_paper],
                                                            mask_vals = mask_vals[on_paper])
        result["prompt_over_text"] = _compute_subset_metrics(bg_all = bg_all[over_text],
                                                             comp_all = comp_all[over_text],
                                                             mask_vals = mask_vals[over_text])
    return result

def save_metrics_json(json_path: str | Path, payload: dict[str, object]) -> Path:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents = True, exist_ok = True)
    json_path.write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
    return json_path