from __future__ import annotations
from pathlib import Path
from src.crops import SHARED_CROPS
from src.ss_simulator.crop import apply_crop
from src.ss_simulator.encode import save_png
from src.ss_simulator.records import build_screenshot_record, save_record
from src.ss_simulator.renderers.pdfium import PdfiumRenderer
import sys

STAGE1_DPI = 96.0

def run_stage1_screenshot(pdf_path: str | Path, output_dir: str | Path, page_index: int = 0, dpi: float = STAGE1_DPI) -> tuple[Path, Path]:
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    renderer = PdfiumRenderer()
    rendered = renderer.render(pdf_path = pdf_path, page_index = page_index, dpi = dpi)
    cropped, crop_info = apply_crop(image = rendered.image, spec = SHARED_CROPS["whole_page"])

    stem = f"{pdf_path.stem}_p{page_index + 1:03d}_{renderer.name}_{int(round(dpi))}dpi"
    png_path, encode_info = save_png(image = cropped, output_path = output_dir / f"{stem}.png", dpi = dpi)
    record = build_screenshot_record(pdf_path = pdf_path, image_path = png_path, render_result = rendered,
                                     crop_info = crop_info, encode_info = encode_info)
    json_path = save_record(record = record, json_path = output_dir / f"{stem}.json")
    return png_path, json_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit()
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "sim_stage1"
    png_out, json_out = run_stage1_screenshot(pdf_path = sys.argv[1], output_dir = out_dir)