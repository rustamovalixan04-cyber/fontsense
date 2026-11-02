"""Higher-resolution, image-only patches shared by training and inference."""
import numpy as np
from PIL import Image, ImageOps
from api.compact import text_crop


def text_patches(image, count=3):
    gray = text_crop(image).convert("L")
    bounds = ImageOps.invert(gray).point(lambda x: 255 if x > 40 else 0).getbbox()
    if bounds is None:
        raise ValueError("No visible text")
    gray = gray.crop(bounds)
    width = max(1, round(gray.width * 48 / gray.height))
    gray = gray.resize((width, 48), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (max(width + 16, 160), 64), 255)
    canvas.paste(gray, ((canvas.width - width) // 2, 8))
    starts = np.linspace(0, canvas.width - 160, count).round().astype(int)
    return np.stack([np.asarray(canvas.crop((int(x), 0, int(x) + 160, 64))) for x in starts])
