from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np

POINTS_PER_INCH = 72.0
DEFAULT_PAGE_HEIGHT_PT = 792.0

FONTS_DIR = Path(__file__).resolve().parents[2] / "fonts"

ATKINSON_FONT_CANDIDATES = (
    FONTS_DIR / "AtkinsonHyperlegibleNext-ExtraBold.ttf",
    FONTS_DIR / "AtkinsonHyperlegibleNext-Bold.ttf",
    FONTS_DIR / "AtkinsonHyperlegibleNext-ExtraBold.otf",
    FONTS_DIR / "AtkinsonHyperlegibleNext-Bold.otf",
)

DEFAULT_TARGET_MODEL_PX = 24.0
DEFAULT_MIN_MODEL_PAGE_HEIGHT_PX = 768.0
DEFAULT_MARGIN_PT = 54.0

DEFAULT_SPACING_X_PT = 24.0
DEFAULT_SPACING_Y_PT = 24.0
DEFAULT_ROW_OFFSET_FRAC = 0.5
DEFAULT_ROTATION_DEG = 0.0
DEFAULT_MAX_TILE_WIDTH_FRAC = 0.5
DEFAULT_MAX_TILE_HEIGHT_FRAC = 0.5
DEFAULT_LINE_SPACING_RATIO = 0.25
DEFAULT_JITTER_X_PT = 0.0
DEFAULT_JITTER_Y_PT = 0.0


@dataclass(frozen = True)
class TileConfig:
    spacing_x_pt: float = DEFAULT_SPACING_X_PT
    spacing_y_pt: float = DEFAULT_SPACING_Y_PT
    row_offset_frac: float = DEFAULT_ROW_OFFSET_FRAC
    rotation_deg: float = DEFAULT_ROTATION_DEG
    max_tile_width_frac: float = DEFAULT_MAX_TILE_WIDTH_FRAC
    max_tile_height_frac: float = DEFAULT_MAX_TILE_HEIGHT_FRAC
    line_spacing_ratio: float = DEFAULT_LINE_SPACING_RATIO
    jitter_x_pt: float = DEFAULT_JITTER_X_PT
    jitter_y_pt: float = DEFAULT_JITTER_Y_PT
    jitter_seed: int | None = None


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


def build_prompt_tile(prompt: str, page_width_px: int, page_height_px: int, dpi: int = 150,
                      font_size_pt: float | None = None, font_path: str | Path | None = None,
                      config: TileConfig | None = None) -> Image.Image:
    if config is None:
        config = TileConfig()
    if config.spacing_x_pt < 0 or config.spacing_y_pt < 0:
        raise ValueError()
    if not (0.0 < config.max_tile_width_frac <= 1.0) or not (0.0 < config.max_tile_height_frac <= 1.0):
        raise ValueError()
    if font_size_pt is None:
        font_size_pt = compute_font_size_pt()

    font_size_px = max(1, pt_to_px(points = font_size_pt, dpi = dpi))
    spacing_x_px = pt_to_px(points = config.spacing_x_pt, dpi = dpi)
    spacing_y_px = pt_to_px(points = config.spacing_y_pt, dpi = dpi)
    max_tile_w = int(page_width_px * config.max_tile_width_frac)
    max_tile_h = int(page_height_px * config.max_tile_height_frac)
    max_text_w = max_tile_w - spacing_x_px
    if max_text_w <= 0:
        raise ValueError()

    line_spacing_px = max(1, int(round(font_size_px * config.line_spacing_ratio)))
    font = load_font(font_size = font_size_px, font_path = font_path)
    wrapped = wrap_prompt_text(prompt = prompt, font = font, max_width_px = max_text_w)

    probe = ImageDraw.Draw(Image.new("L", (1, 1), color = 0))
    bbox = probe.multiline_textbbox((0, 0), wrapped, font = font, align = "center", spacing = line_spacing_px)
    block_w = max(1, bbox[2] - bbox[0])
    block_h = max(1, bbox[3] - bbox[1])
    block_w = max(1, int(math.ceil(block_w)))
    block_h = max(1, int(math.ceil(block_h)))

    block = Image.new("L", (block_w, block_h), color = 0)
    draw = ImageDraw.Draw(block)
    draw.multiline_text((-bbox[0], -bbox[1]), wrapped, fill = 255, font = font, align = "center",
                        spacing = line_spacing_px)

    if config.rotation_deg != 0.0:
        block = block.rotate(config.rotation_deg, resample = Image.Resampling.BICUBIC, expand = True, fillcolor = 0)

    tile_w = block.width + spacing_x_px
    tile_h = block.height + spacing_y_px
    if tile_w > max_tile_w or tile_h > max_tile_h:
        raise ValueError()

    tile = Image.new("L", (tile_w, tile_h), color = 0)
    tile.paste(block, (spacing_x_px // 2, spacing_y_px // 2))
    return tile


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


def render_tiled_prompt_mask(prompt: str, width_px: int, height_px: int, dpi: int = 150,
                             font_size_pt: float | None = None, font_path: str | Path | None = None,
                             config: TileConfig | None = None) -> Image.Image:
    if config is None:
        config = TileConfig()
    if config.jitter_x_pt < 0 or config.jitter_y_pt < 0:
        raise ValueError()
    if (config.jitter_x_pt > 0 or config.jitter_y_pt > 0) and config.jitter_seed is None:
        raise ValueError()

    canvas = np.zeros((height_px, width_px), dtype = np.uint8)
    if not prompt or not prompt.strip():
        return Image.fromarray(canvas, mode = "L")

    tile_img = build_prompt_tile(prompt = prompt, page_width_px = width_px, page_height_px = height_px, dpi = dpi,
                                 font_size_pt = font_size_pt, font_path = font_path, config = config)
    tile_arr = np.asarray(tile_img, dtype = np.uint8)
    tile_h, tile_w = tile_arr.shape

    jitter_x_px = pt_to_px(points = config.jitter_x_pt, dpi = dpi)
    jitter_y_px = pt_to_px(points = config.jitter_y_pt, dpi = dpi)
    rng = np.random.default_rng(config.jitter_seed) if (jitter_x_px > 0 or jitter_y_px > 0) else None

    start_row = -1 if jitter_y_px > 0 else 0
    row = start_row
    while row * tile_h - jitter_y_px < height_px:
        base_y = row * tile_h
        row_shift_px = int(round(((row * config.row_offset_frac) % 1.0) * tile_w))
        start_x = row_shift_px - tile_w
        if jitter_x_px > 0:
            start_x -= tile_w

        base_x = start_x
        while base_x - jitter_x_px < width_px:
            dx = int(rng.integers(-jitter_x_px, jitter_x_px + 1)) if (rng is not None and jitter_x_px > 0) else 0
            dy = int(rng.integers(-jitter_y_px, jitter_y_px + 1)) if (rng is not None and jitter_y_px > 0) else 0
            _stamp_tile(canvas = canvas, tile = tile_arr, x = base_x + dx, y = base_y + dy)
            base_x += tile_w
        row += 1

    return Image.fromarray(canvas, mode = "L")