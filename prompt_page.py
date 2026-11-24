from __future__ import annotations
import numpy as np
import pymupdf
import io
import sys
import warnings
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# letter size (8.5 × 11) * 72 in pdf points:
PAGE_WIDTH_PT = 612.0
PAGE_HEIGHT_PT = 792.0

ATKINSON_FONT_CANDIDATES = (
    "AtkinsonHyperlegibleNext-ExtraBold.ttf",
    "AtkinsonHyperlegibleNext-Bold.ttf",
    "AtkinsonHyperlegibleNext-ExtraBold.otf",
    "AtkinsonHyperlegibleNext-Bold.otf")

def load_font(font_size: int, font_path: str | Path | None = None) -> ImageFont.FreeTypeFont:
    candidates: tuple[str | Path, ...]
    if font_path is not None:
        candidates = (font_path,)
    else:
        candidates = ATKINSON_FONT_CANDIDATES

    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=font_size)
        except OSError:
            continue

    raise FileNotFoundError()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt_arg = sys.argv[1]
    else:
        prompt_arg = "default"