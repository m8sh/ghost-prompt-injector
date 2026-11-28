from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import math
import numpy as np
from src.crops import CropName, SHARED_CROPS, compute_tight_text_box

POINTS_PER_INCH = 72.0
DEFAULT_PAGE_HEIGHT_PT = 792.0

FONTS_DIR = Path(__file__).resolve().parents[2] / "fonts"

ATKINSON_FONT_CANDIDATES = (
    FONTS_DIR / "AtkinsonHyperlegibleNext-ExtraBold.ttf",
    FONTS_DIR / "AtkinsonHyperlegibleNext-Bold.ttf",
    FONTS_DIR / "AtkinsonHyperlegibleNext-ExtraBold.otf",
    FONTS_DIR / "AtkinsonHyperlegibleNext-Bold.otf",
)

PlacementMode = Literal["under", "avoid", "hybrid"]

DEFAULT_TARGET_MODEL_PX = 24.0
DEFAULT_MIN_MODEL_PAGE_HEIGHT_PX = 768.0
DEFAULT_MARGIN_PT = 54.0

DEFAULT_SPACING_X_PT = 24.0
DEFAULT_SPACING_Y_PT = 24.0
DEFAULT_ROW_OFFSET_FRAC = 0.5
DEFAULT_ROTATION_DEG = 0.0
DEFAULT_TARGET_CROP: CropName = "half_page"
DEFAULT_LINE_SPACING_RATIO = 0.25
DEFAULT_JITTER_X_PT = 0.0
DEFAULT_JITTER_Y_PT = 0.0


@dataclass(frozen = True)
class TileConfig:
    spacing_x_pt: float = DEFAULT_SPACING_X_PT
    spacing_y_pt: float = DEFAULT_SPACING_Y_PT
    row_offset_frac: float = DEFAULT_ROW_OFFSET_FRAC
    rotation_deg: float = DEFAULT_ROTATION_DEG
    target_crop: CropName = DEFAULT_TARGET_CROP
    line_spacing_ratio: float = DEFAULT_LINE_SPACING_RATIO
    jitter_x_pt: float = DEFAULT_JITTER_X_PT
    jitter_y_pt: float = DEFAULT_JITTER_Y_PT
    jitter_seed: int | None = None


@dataclass(frozen = True)
class TileLayout:
    mask: Image.Image
    copy_boxes: tuple[tuple[int, int, int, int], ...]
    intact_boxes: tuple[tuple[int, int, int, int], ...]
    y_offset_px: int
    font_size_pt: float
    tile_size_px: tuple[int, int]
    block_size_px: tuple[int, int]


def pt_to_px(points: float, dpi: int) -> int:
    return int(round(points * dpi / POINTS_PER_INCH))


def load_font(font_size: int, font_path: str | Path | None = None) -> ImageFont.FreeTypeFont:
    candidates: tuple[str | Path, ...]
    if font_path is not None:
        candidates = (font_path,)
    else:
        candidates = ATKINSON_FONT_CANDIDATES

    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size = font_size)
        except OSError:
            continue

    raise FileNotFoundError("valid font not found")


def compute_font_size_pt(target_model_px: float = DEFAULT_TARGET_MODEL_PX,
                         min_model_page_height_px: float = DEFAULT_MIN_MODEL_PAGE_HEIGHT_PX,
                         page_height_pt: float = DEFAULT_PAGE_HEIGHT_PT) -> float:
    if target_model_px <= 0 or min_model_page_height_px <= 0:
        raise ValueError()
    effective_px_per_pt = min_model_page_height_px / page_height_pt
    return target_model_px / effective_px_per_pt


def _measure_line_width_px(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> int:
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font = font)
    return bbox[2] - bbox[0]


def wrap_prompt_text(prompt: str, font: ImageFont.FreeTypeFont, max_width_px: int) -> str:
    if max_width_px <= 0:
        raise ValueError("max_width_px isn't positive")

    probe = ImageDraw.Draw(Image.new("L", (1, 1), color = 0))
    wrapped_lines: list[str] = []

    for paragraph in prompt.splitlines():
        words = paragraph.split()
        if not words:
            wrapped_lines.append("")
            continue

        current_line = ""
        for word in words:
            if not current_line:
                candidate = word
            else:
                candidate = f"{current_line} {word}"

            if _measure_line_width_px(draw = probe, text = candidate, font = font) <= max_width_px:
                current_line = candidate
                continue

            if current_line:
                wrapped_lines.append(current_line)
                current_line = ""

            if _measure_line_width_px(draw = probe, text = word, font = font) <= max_width_px:
                current_line = word
            else:
                chunk = ""
                for char in word:
                    next_chunk = chunk + char
                    if chunk and _measure_line_width_px(draw = probe, text = next_chunk, font = font) > max_width_px:
                        wrapped_lines.append(chunk)
                        chunk = char
                    else:
                        chunk = next_chunk
                current_line = chunk

        if current_line:
            wrapped_lines.append(current_line)

    return "\n".join(wrapped_lines)


def render_prompt_mask(prompt: str, width_px: int, height_px: int, dpi: int = 150,
                       font_size_pt: float | None = None, margin_pt: float = DEFAULT_MARGIN_PT,
                       font_path: str | Path | None = None) -> Image.Image:
    mask = Image.new("L", (width_px, height_px), color = 0)
    if not prompt or not prompt.strip():
        return mask

    if font_size_pt is None:
        font_size_pt = compute_font_size_pt()

    font_size_px = max(1, pt_to_px(points = font_size_pt, dpi = dpi))
    margin_px = pt_to_px(points = margin_pt, dpi = dpi)
    max_width_px = width_px - 2 * margin_px
    max_height_px = height_px - 2 * margin_px
    spacing_px = max(4, int(round(font_size_px * DEFAULT_LINE_SPACING_RATIO)))

    font = load_font(font_size = font_size_px, font_path = font_path)
    wrapped = wrap_prompt_text(prompt = prompt, font = font, max_width_px = max_width_px)

    draw = ImageDraw.Draw(mask)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font = font, align = "center", spacing = spacing_px)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    if text_h > max_height_px:
        raise ValueError()

    x = (width_px - text_w) / 2 - bbox[0]
    y = (height_px - text_h) / 2 - bbox[1]

    draw.multiline_text((x, y), wrapped, fill = 255, font = font, align = "center", spacing = spacing_px)
    return mask


def _resolve_crop_bounds_px(page_width_px: int, page_height_px: int, target_crop: CropName) -> tuple[int, int]:
    if target_crop not in SHARED_CROPS:
        raise ValueError()
    spec = SHARED_CROPS[target_crop]
    crop_w = int(page_width_px * spec.width_frac) if spec.width_frac is not None else page_width_px
    crop_h = int(page_height_px * spec.height_frac) if spec.height_frac is not None else page_height_px
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError()
    return crop_w, crop_h


def _build_prompt_tile_parts(prompt: str, page_width_px: int, page_height_px: int, dpi: int = 150,
                             font_size_pt: float | None = None, font_path: str | Path | None = None,
                             config: TileConfig | None = None) -> tuple[Image.Image, Image.Image, int, int, float]:
    if config is None:
        config = TileConfig()
    if config.spacing_x_pt < 0 or config.spacing_y_pt < 0:
        raise ValueError()
    if font_size_pt is None:
        font_size_pt = compute_font_size_pt()

    font_size_px = max(1, pt_to_px(points = font_size_pt, dpi = dpi))
    spacing_x_px = pt_to_px(points = config.spacing_x_pt, dpi = dpi)
    spacing_y_px = pt_to_px(points = config.spacing_y_pt, dpi = dpi)
    crop_w, crop_h = _resolve_crop_bounds_px(page_width_px = page_width_px, page_height_px = page_height_px,
                                             target_crop = config.target_crop)
    max_text_w = (crop_w - spacing_x_px) // 2
    if max_text_w <= 0:
        raise ValueError()

    line_spacing_px = max(1, int(round(font_size_px * config.line_spacing_ratio)))
    font = load_font(font_size = font_size_px, font_path = font_path)
    wrapped = wrap_prompt_text(prompt = prompt, font = font, max_width_px = max_text_w)

    probe = ImageDraw.Draw(Image.new("L", (1, 1), color = 0))
    bbox = probe.multiline_textbbox((0, 0), wrapped, font = font, align = "center", spacing = line_spacing_px)
    block_w = max(1, int(math.ceil(bbox[2] - bbox[0])))
    block_h = max(1, int(math.ceil(bbox[3] - bbox[1])))

    block = Image.new("L", (block_w, block_h), color = 0)
    draw = ImageDraw.Draw(block)
    draw.multiline_text((-bbox[0], -bbox[1]), wrapped, fill = 255, font = font, align = "center",
                        spacing = line_spacing_px)

    if config.rotation_deg != 0.0:
        block = block.rotate(config.rotation_deg, resample = Image.Resampling.BICUBIC, expand = True, fillcolor = 0)

    tile_w = block.width + spacing_x_px
    tile_h = block.height + spacing_y_px
    if tile_w + block.width > crop_w or tile_h + block.height > crop_h:
        raise ValueError()

    pad_x = spacing_x_px // 2
    pad_y = spacing_y_px // 2
    tile = Image.new("L", (tile_w, tile_h), color = 0)
    tile.paste(block, (pad_x, pad_y))
    return tile, block, pad_x, pad_y, font_size_pt


def build_prompt_tile(prompt: str, page_width_px: int, page_height_px: int, dpi: int = 150,
                      font_size_pt: float | None = None, font_path: str | Path | None = None,
                      config: TileConfig | None = None) -> Image.Image:
    tile, _, _, _, _ = _build_prompt_tile_parts(prompt = prompt, page_width_px = page_width_px,
                                                page_height_px = page_height_px, dpi = dpi,
                                                font_size_pt = font_size_pt, font_path = font_path, config = config)
    return tile


def _row_vertical_clearance(by0: int, by1: int, content_ys: np.ndarray, height_px: int) -> int:
    above = content_ys[content_ys < by0]
    below = content_ys[content_ys >= by1]
    dist_above = int(by0 - above.max()) if above.size > 0 else height_px
    dist_below = int(below.min() - by1) if below.size > 0 else height_px
    return min(dist_above, dist_below)


def find_hybrid_y_offset(block_ink: np.ndarray, tile_w: int, tile_h: int, pad_x: int, pad_y: int,
                         content_map: np.ndarray, row_offset_frac: float) -> int:
    height_px, width_px = content_map.shape
    row_has_content = content_map.any(axis = 1)
    if not row_has_content.any():
        return 0

    block_h, block_w = block_ink.shape
    content_ys = np.flatnonzero(row_has_content)

    best_offset = 0
    best_score = (-1, -1, -1, -1)

    for y_offset in range(tile_h):
        intact_count = 0
        erased_px = 0
        min_clearance = height_px

        row = -1 if y_offset > 0 else 0
        while row * tile_h + y_offset < height_px:
            base_y = row * tile_h + y_offset
            by0 = base_y + pad_y
            by1 = by0 + block_h
            row_shift_px = int(round(((row * row_offset_frac) % 1.0) * tile_w))
            row += 1

            if by1 <= 0 or by0 >= height_px:
                continue

            cy0 = max(0, by0)
            cy1 = min(height_px, by1)
            row_in_bounds = (by0 >= 0) and (by1 <= height_px)
            band_has_content = bool(row_has_content[cy0:cy1].any())

            base_x = row_shift_px - tile_w
            if not band_has_content:
                if row_in_bounds:
                    clearance = _row_vertical_clearance(by0 = by0, by1 = by1, content_ys = content_ys,
                                                        height_px = height_px)
                    min_clearance = min(min_clearance, clearance)
                    while base_x < width_px:
                        bx0 = base_x + pad_x
                        if bx0 >= 0 and (bx0 + block_w) <= width_px:
                            intact_count += 1
                        base_x += tile_w
                continue

            while base_x < width_px:
                bx0 = base_x + pad_x
                bx1 = bx0 + block_w
                if bx1 > 0 and bx0 < width_px:
                    cx0 = max(0, bx0)
                    cx1 = min(width_px, bx1)
                    overlap = int(np.count_nonzero(
                        content_map[cy0:cy1, cx0:cx1] & block_ink[cy0 - by0:cy1 - by0, cx0 - bx0:cx1 - bx0]))
                    erased_px += overlap
                    if row_in_bounds and bx0 >= 0 and bx1 <= width_px and overlap == 0:
                        intact_count += 1
                base_x += tile_w

        score = (intact_count, -erased_px, min_clearance, -y_offset)
        if score > best_score:
            best_score = score
            best_offset = y_offset

    return best_offset


def _stamp_tile(canvas: np.ndarray, tile: np.ndarray, x: int, y: int) -> None:
    canvas_h, canvas_w = canvas.shape
    tile_h, tile_w = tile.shape

    dst_x0 = max(0, x)
    dst_y0 = max(0, y)
    dst_x1 = min(canvas_w, x + tile_w)
    dst_y1 = min(canvas_h, y + tile_h)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return

    src_x0 = dst_x0 - x
    src_y0 = dst_y0 - y
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    np.maximum(canvas[dst_y0:dst_y1, dst_x0:dst_x1], tile[src_y0:src_y1, src_x0:src_x1],
               out = canvas[dst_y0:dst_y1, dst_x0:dst_x1])


def render_tiled_prompt_layout(prompt: str, width_px: int, height_px: int, dpi: int = 150,
                               font_size_pt: float | None = None, font_path: str | Path | None = None,
                               config: TileConfig | None = None, placement_mode: PlacementMode = "avoid",
                               keepout_mask: np.ndarray | None = None,
                               content_map: np.ndarray | None = None) -> TileLayout:
    if config is None:
        config = TileConfig()
    if config.jitter_x_pt < 0 or config.jitter_y_pt < 0:
        raise ValueError()
    if (config.jitter_x_pt > 0 or config.jitter_y_pt > 0) and config.jitter_seed is None:
        raise ValueError()
    if placement_mode not in ("under", "avoid", "hybrid"):
        raise ValueError()

    resolved_font_size_pt = font_size_pt if font_size_pt is not None else compute_font_size_pt()
    canvas = np.zeros((height_px, width_px), dtype = np.uint8)
    if not prompt or not prompt.strip():
        return TileLayout(mask = Image.fromarray(canvas, mode = "L"), copy_boxes = (), intact_boxes = (),
                          y_offset_px = 0, font_size_pt = resolved_font_size_pt, tile_size_px = (0, 0),
                          block_size_px = (0, 0))

    tile_img, block_img, pad_x, pad_y, resolved_font_size_pt = _build_prompt_tile_parts(
        prompt = prompt, page_width_px = width_px, page_height_px = height_px, dpi = dpi,
        font_size_pt = resolved_font_size_pt, font_path = font_path, config = config)
    tile_arr = np.asarray(tile_img, dtype = np.uint8)
    block_ink = np.asarray(block_img, dtype = np.uint8) > 0
    tile_h, tile_w = tile_arr.shape
    block_h, block_w = block_ink.shape

    keepout_bool: np.ndarray | None = None
    if keepout_mask is not None:
        keepout_bool = np.asarray(keepout_mask, dtype = bool)
        if keepout_bool.shape != (height_px, width_px):
            raise ValueError()

    shift_map: np.ndarray | None = None
    if placement_mode == "hybrid":
        source_map = content_map if content_map is not None else keepout_bool
        if source_map is not None:
            shift_map = np.asarray(source_map, dtype = bool)
            if shift_map.shape != (height_px, width_px):
                raise ValueError()

    y_offset_px = 0
    if shift_map is not None:
        y_offset_px = find_hybrid_y_offset(block_ink = block_ink, tile_w = tile_w, tile_h = tile_h, pad_x = pad_x,
                                           pad_y = pad_y, content_map = shift_map,
                                           row_offset_frac = config.row_offset_frac)

    jitter_x_px = pt_to_px(points = config.jitter_x_pt, dpi = dpi)
    jitter_y_px = pt_to_px(points = config.jitter_y_pt, dpi = dpi)
    rng = np.random.default_rng(config.jitter_seed) if (jitter_x_px > 0 or jitter_y_px > 0) else None

    copy_boxes: list[tuple[int, int, int, int]] = []
    intact_boxes: list[tuple[int, int, int, int]] = []

    start_row = -1 if (y_offset_px > 0 or jitter_y_px > 0) else 0
    row = start_row
    while row * tile_h + y_offset_px - jitter_y_px < height_px:
        base_y = row * tile_h + y_offset_px
        row_shift_px = int(round(((row * config.row_offset_frac) % 1.0) * tile_w))
        start_x = row_shift_px - tile_w
        if jitter_x_px > 0:
            start_x -= tile_w

        base_x = start_x
        while base_x - jitter_x_px < width_px:
            dx = int(rng.integers(-jitter_x_px, jitter_x_px + 1)) if (rng is not None and jitter_x_px > 0) else 0
            dy = int(rng.integers(-jitter_y_px, jitter_y_px + 1)) if (rng is not None and jitter_y_px > 0) else 0
            x = base_x + dx
            y = base_y + dy
            _stamp_tile(canvas = canvas, tile = tile_arr, x = x, y = y)

            bx0 = x + pad_x
            by0 = y + pad_y
            bx1 = bx0 + block_w
            by1 = by0 + block_h
            if bx0 >= 0 and by0 >= 0 and bx1 <= width_px and by1 <= height_px:
                copy_boxes.append((bx0, by0, bx1, by1))
                if keepout_bool is None or not bool(np.any(keepout_bool[by0:by1, bx0:bx1] & block_ink)):
                    intact_boxes.append((bx0, by0, bx1, by1))

            base_x += tile_w
        row += 1

    if keepout_bool is not None:
        canvas[keepout_bool] = 0

    return TileLayout(mask = Image.fromarray(canvas, mode = "L"), copy_boxes = tuple(copy_boxes),
                      intact_boxes = tuple(intact_boxes), y_offset_px = y_offset_px,
                      font_size_pt = resolved_font_size_pt, tile_size_px = (tile_w, tile_h),
                      block_size_px = (block_w, block_h))


def render_tiled_prompt_mask(prompt: str, width_px: int, height_px: int, dpi: int = 150,
                             font_size_pt: float | None = None, font_path: str | Path | None = None,
                             config: TileConfig | None = None, placement_mode: PlacementMode = "avoid",
                             keepout_mask: np.ndarray | None = None,
                             content_map: np.ndarray | None = None) -> Image.Image:
    layout = render_tiled_prompt_layout(prompt = prompt, width_px = width_px, height_px = height_px, dpi = dpi,
                                        font_size_pt = font_size_pt, font_path = font_path, config = config,
                                        placement_mode = placement_mode, keepout_mask = keepout_mask,
                                        content_map = content_map)
    return layout.mask


def _sliding_window_coverage(intact_boxes: tuple[tuple[int, int, int, int], ...], width_px: int, height_px: int,
                             crop_w: int, crop_h: int) -> float:
    if not intact_boxes or crop_w <= 0 or crop_h <= 0:
        return 0.0
    max_x = max(0, width_px - crop_w)
    max_y = max(0, height_px - crop_h)
    covered = np.zeros((max_y + 1, max_x + 1), dtype = bool)
    for bx0, by0, bx1, by1 in intact_boxes:
        ox0 = max(0, bx1 - crop_w)
        ox1 = min(max_x, bx0)
        oy0 = max(0, by1 - crop_h)
        oy1 = min(max_y, by0)
        if ox0 <= ox1 and oy0 <= oy1:
            covered[oy0:oy1 + 1, ox0:ox1 + 1] = True
    return round(float(covered.mean()), 4)


def compute_crop_coverage(intact_boxes: tuple[tuple[int, int, int, int], ...], width_px: int, height_px: int,
                          dpi: int, content_map: np.ndarray | None = None) -> dict[str, float | None]:
    results: dict[str, float | None] = {}
    for crop_name, spec in SHARED_CROPS.items():
        if crop_name == "tight_text":
            if content_map is None:
                results[crop_name] = None
                continue
            tight_box = compute_tight_text_box(content_map = content_map, dpi = dpi, padding_pt = spec.padding_pt)
            if tight_box is None:
                results[crop_name] = None
                continue
            tx0, ty0, tx1, ty1 = tight_box
            has_intact = any(bx0 >= tx0 and by0 >= ty0 and bx1 <= tx1 and by1 <= ty1 for bx0, by0, bx1,
            by1 in intact_boxes)
            results[crop_name] = 1.0 if has_intact else 0.0
            continue

        crop_w = int(width_px * spec.width_frac) if spec.width_frac is not None else width_px
        crop_h = int(height_px * spec.height_frac) if spec.height_frac is not None else height_px
        results[crop_name] = _sliding_window_coverage(intact_boxes = intact_boxes, width_px = width_px,
                                                      height_px = height_px, crop_w = crop_w, crop_h = crop_h)
    return results