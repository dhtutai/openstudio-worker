"""One-click upscaling with Real-ESRGAN x4plus (BSD-3-Clause weights, loaded with the MIT-licensed spandrel)."""
import numpy as np
import torch
from PIL import Image

TARGET_LONG_SIDE = 3840  # 4K
MAX_INPUT_PIXELS = 4_500_000  # Covers every 2K output; keeps the 4x intermediate image within worker memory.
MIN_SIDE = 16
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
WEIGHTS_SHA256 = "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"


def target_size(width, height):
    """Long side to 4K, keeping the shape. The model enlarges at most 4x, so small photos stop short of 4K."""
    if max(width, height) >= TARGET_LONG_SIDE:
        raise ValueError("This image is already 4K or larger.")
    if width * height > MAX_INPUT_PIXELS:
        raise ValueError("This image is too large to upscale. Use one of about 4 megapixels or less.")
    if min(width, height) < MIN_SIDE:
        raise ValueError("This image is too small to upscale.")
    scale = min(4.0, TARGET_LONG_SIDE / max(width, height))
    return round(width * scale), round(height * scale)


def upscale(model, image, device="cuda", tile=512, overlap=32):
    """Run a 4x model tile by tile. Each tile gets surrounding context that is then cut off, to hide seams."""
    size = target_size(*image.size)
    source = np.asarray(image.convert("RGB"))
    height, width = source.shape[:2]
    result = np.empty((height * 4, width * 4, 3), dtype=np.uint8)
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            bottom, right = min(top + tile, height), min(left + tile, width)
            y0, x0 = max(top - overlap, 0), max(left - overlap, 0)
            y1, x1 = min(bottom + overlap, height), min(right + overlap, width)
            patch = torch.from_numpy(source[y0:y1, x0:x1].copy()).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
            with torch.inference_mode():
                out = model(patch).clamp(0, 1)[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
            result[top * 4:bottom * 4, left * 4:right * 4] = out[(top - y0) * 4:(bottom - y0) * 4, (left - x0) * 4:(right - x0) * 4]
    return Image.fromarray(result).resize(size, Image.LANCZOS)
