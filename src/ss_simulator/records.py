from __future__ import annotations
from pathlib import Path
from src.ss_simulator.renderers.base import RenderResult
import json
import platform
import sys

def build_screenshot_record(pdf_path: str | Path, image_path: str | Path, render_result: RenderResult, crop_info: dict[str, object], encode_info: dict[str, object]) -> dict[str, object]:
    return {
        "source_pdf": str(pdf_path),
        "output_image": str(image_path),
        "page_index": render_result.page_index,
        "dpi": render_result.dpi,
        "renderer": {
            "name": render_result.renderer_name,
            "version": render_result.renderer_version,
            "engine": render_result.engine_name,
            "engine_version": render_result.engine_version,
            "details": render_result.details,
        },
        "crop": crop_info,
        "encode": encode_info,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }

def save_record(record: dict[str, object], json_path: str | Path) -> Path:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents = True, exist_ok = True)
    json_path.write_text(json.dumps(record, indent = 2), encoding = "utf-8")
    return json_path