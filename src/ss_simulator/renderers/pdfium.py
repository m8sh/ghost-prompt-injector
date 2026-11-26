from __future__ import annotations
from pathlib import Path
from src.ss_simulator.renderers.base import RenderResult
import numpy as np
import pypdfium2 as pdfium

POINTS_PER_INCH = 72.0

class PdfiumRenderer:
    name: str = "pdfium"

    def render(self, pdf_path: str | Path, page_index: int, dpi: float) -> RenderResult:
        if dpi <= 0:
            raise ValueError()

        pdf_path = Path(pdf_path)
        scale = dpi / POINTS_PER_INCH
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            if not (0 <= page_index < len(doc)):
                raise IndexError()

            page = doc[page_index]
            try:
                page_width_pt, page_height_pt = page.get_size()
                bitmap = page.render(scale = scale, rotation = 0, rev_byteorder = True)
                try:
                    rgb = np.asarray(bitmap.to_pil().convert("RGB"), dtype = np.uint8).copy()
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            doc.close()

        return RenderResult(
            image = rgb,
            renderer_name = self.name,
            renderer_version = str(pdfium.PYPDFIUM_INFO.version),
            engine_name = "pdfium",
            engine_version = str(pdfium.PDFIUM_INFO.version),
            dpi = float(dpi),
            page_index = page_index,
            details = {
                "scale": scale,
                "pdfium_build": str(pdfium.PDFIUM_INFO.build),
                "page_size_pt": [round(float(page_width_pt), 2), round(float(page_height_pt), 2)],
                "image_size_px": [int(rgb.shape[1]), int(rgb.shape[0])],
            },
        )