from __future__ import annotations
from PIL import Image
from pathlib import Path
import numpy as np

def save_png(image: np.ndarray, output_path: str | Path, dpi: float) -> tuple[Path, dict[str, object]]:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)
    Image.fromarray(image, mode = "RGB").save(output_path, format = "PNG", dpi = (dpi, dpi))
    return output_path, {"format": "PNG", "lossless": True, "size_bytes": output_path.stat().st_size}