from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

@dataclass(frozen = True)
class RenderResult:
    image: np.ndarray
    renderer_name: str
    renderer_version: str
    engine_name: str
    engine_version: str
    dpi: float
    page_index: int
    details: dict[str, object] = field(default_factory = dict)

class PdfRenderer(Protocol):
    name: str

    def render(self, pdf_path: str | Path, page_index: int, dpi: float) -> RenderResult:
        pass