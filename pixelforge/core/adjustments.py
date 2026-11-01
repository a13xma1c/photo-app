"""Non-destructive slider adjustments (exposure, contrast, colour, effects...).

Every adjustment is described by a :class:`SliderSpec` in :data:`SLIDERS`.  The
values of all sliders are stored in a plain ``dict[str, int]`` so they can be
serialised trivially and compared cheaply.  :func:`apply_adjustments` turns an
image plus such a dict into a new image.

Rendering is *scale aware*: the GUI renders downscaled previews, and
size-dependent effects (blur, sharpen, vignette) must look identical in the
preview and in the full-size export.  Callers therefore pass the document size
and the preview scale so radii can be computed in document space.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class SliderSpec:
    label: str
    min: int
    max: int
    default: int
    group: str
    tooltip: str = ""


SLIDERS: dict[str, SliderSpec] = {
    # --- light ------------------------------------------------------------
    "exposure": SliderSpec("Exposure", -100, 100, 0, "Light", "Overall exposure in stops (±2 EV)"),
    "brightness": SliderSpec("Brightness", -100, 100, 0, "Light", "Linear brightness"),
    "contrast": SliderSpec("Contrast", -100, 100, 0, "Light", "Contrast around mid grey"),
    "highlights": SliderSpec("Highlights", -100, 100, 0, "Light", "Recover or boost bright areas"),
    "shadows": SliderSpec("Shadows", -100, 100, 0, "Light", "Lift or deepen dark areas"),
    "gamma": SliderSpec("Gamma", -100, 100, 0, "Light", "Mid-tone gamma curve"),
    # --- colour -----------------------------------------------------------
    "saturation": SliderSpec("Saturation", -100, 100, 0, "Color", "Colour intensity"),
    "vibrance": SliderSpec("Vibrance", -100, 100, 0, "Color", "Boost muted colours more than saturated ones"),
    "temperature": SliderSpec("Temperature", -100, 100, 0, "Color", "Cool (blue) to warm (orange)"),
    "tint": SliderSpec("Tint", -100, 100, 0, "Color", "Green to magenta"),
    "hue": SliderSpec("Hue", -180, 180, 0, "Color", "Rotate all hues (degrees)"),
    # --- effects ----------------------------------------------------------
    "sharpen": SliderSpec("Sharpen", 0, 100, 0, "Effects", "Unsharp-mask sharpening"),
    "blur": SliderSpec("Blur", 0, 20, 0, "Effects", "Gaussian blur"),
    "vignette": SliderSpec("Vignette", 0, 100, 0, "Effects", "Darken the corners"),
    "grayscale": SliderSpec("Grayscale", 0, 100, 0, "Effects", "Mix towards black & white"),
    "sepia": SliderSpec("Sepia", 0, 100, 0, "Effects", "Mix towards sepia tone"),
}

GROUPS: tuple[str, ...] = ("Light", "Color", "Effects")

_SEPIA_MATRIX = (
    0.393,
    0.769,
    0.189,
    0,
    0.349,
    0.686,
    0.168,
    0,
    0.272,
    0.534,
    0.131,
    0,
)


def default_adjustments() -> dict[str, int]:
    """A fresh dict with every slider at its neutral value."""
    return {key: spec.default for key, spec in SLIDERS.items()}


def normalize_adjustments(adj: Mapping[str, float] | None) -> dict[str, int]:
    """Fill in missing keys, drop unknown ones and clamp values to the slider range."""
    out = default_adjustments()
    for key, value in (adj or {}).items():
        spec = SLIDERS.get(key)
        if spec is not None:
            out[key] = int(min(spec.max, max(spec.min, round(value))))
    return out


def is_neutral(adj: dict[str, int] | None) -> bool:
    """True when no slider deviates from its default."""
    if not adj:
        return True
    return all(SLIDERS[k].default == v for k, v in adj.items() if k in SLIDERS)


# ---------------------------------------------------------------- helpers --
def _lut(fn) -> list[int]:
    return [int(min(255, max(0, round(fn(i))))) for i in range(256)]


def _point_rgb(img: Image.Image, fn) -> Image.Image:
    """Apply a per-channel LUT built from *fn* to every band of an RGB image."""
    return img.point(_lut(fn) * len(img.getbands()))


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _luma(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def _split_alpha(img: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    if img.mode == "RGBA":
        return img.convert("RGB"), img.getchannel("A")
    if img.mode in ("LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        return rgba.convert("RGB"), rgba.getchannel("A")
    return img.convert("RGB"), None


# -------------------------------------------------------- numpy stage ops --
def _apply_highlights_shadows(arr: np.ndarray, highlights: int, shadows: int) -> np.ndarray:
    luma = _luma(arr)
    if highlights:
        weight = _smoothstep(0.45, 1.0, luma)[..., None]
        amount = highlights / 100 * 0.6
        if amount > 0:
            arr = arr + amount * weight * (1 - arr)
        else:
            arr = arr + amount * weight * arr
    if shadows:
        weight = (1 - _smoothstep(0.0, 0.55, luma))[..., None]
        amount = shadows / 100 * 0.6
        if amount > 0:
            arr = arr + amount * weight * (1 - arr)
        else:
            arr = arr + amount * weight * arr
    return arr


def _apply_vibrance(arr: np.ndarray, vibrance: int) -> np.ndarray:
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    amount = vibrance / 100
    # muted colours are affected more than saturated ones
    boost = amount * (1 - sat) if amount > 0 else amount * np.ones_like(sat)
    mean = arr.mean(axis=2, keepdims=True)
    return mean + (arr - mean) * (1 + boost[..., None])


def _apply_vignette(
    arr: np.ndarray, amount: int, doc_size: tuple[float, float], region: tuple[float, float, float, float], scale: float
) -> np.ndarray:
    h, w = arr.shape[:2]
    doc_w, doc_h = doc_size
    left, top = region[0], region[1]
    xs = left + (np.arange(w, dtype=np.float32) + 0.5) / scale
    ys = top + (np.arange(h, dtype=np.float32) + 0.5) / scale
    nx = (xs - doc_w / 2) / max(doc_w / 2, 1e-6)
    ny = (ys - doc_h / 2) / max(doc_h / 2, 1e-6)
    radius = np.sqrt(nx[None, :] ** 2 + ny[:, None] ** 2) / np.sqrt(2)
    strength = amount / 100
    falloff = _smoothstep(0.25, 1.05, radius) * strength
    return arr * (1 - falloff)[..., None]


# ---------------------------------------------------------------- public --
def apply_adjustments(
    img: Image.Image,
    adj: dict[str, int] | None,
    *,
    doc_size: tuple[float, float] | None = None,
    region: tuple[float, float, float, float] | None = None,
    scale: float = 1.0,
) -> Image.Image:
    """Return a new image with the slider adjustments applied.

    ``img`` may be any mode; the result is RGBA when the input had an alpha
    channel and RGB otherwise.  ``doc_size`` is the size of the full document
    (in document pixels), ``region`` the document-space box that ``img``
    covers and ``scale`` the number of image pixels per document pixel.  They
    default to treating ``img`` as the whole document at 100 %.
    """
    adj = normalize_adjustments(adj)
    rgb, alpha = _split_alpha(img)
    if is_neutral(adj):
        return Image.merge("RGBA", (*rgb.split(), alpha)) if alpha is not None else rgb

    if doc_size is None:
        doc_size = (img.width / scale, img.height / scale)
    if region is None:
        region = (0.0, 0.0, doc_size[0], doc_size[1])
    # Reference length in *image* pixels: effect radii are expressed relative
    # to the document so previews match the full-size export.
    ref = max(doc_size) * scale

    out = rgb
    if adj["blur"] > 0:
        radius = adj["blur"] * ref / 800
        if alpha is not None:
            # blur premultiplied to avoid dark fringes along transparent edges
            blurred = Image.merge("RGBA", (*out.split(), alpha)).convert("RGBa")
            blurred = blurred.filter(ImageFilter.GaussianBlur(radius)).convert("RGBA")
            out, alpha = blurred.convert("RGB"), blurred.getchannel("A")
        else:
            out = out.filter(ImageFilter.GaussianBlur(radius))

    if adj["sharpen"] > 0:
        radius = max(1.0, min(3.0, ref / 1200))
        out = out.filter(ImageFilter.UnsharpMask(radius=radius, percent=adj["sharpen"] * 3, threshold=1))

    if adj["hue"]:
        h, s, v = out.convert("HSV").split()
        shift = round(adj["hue"] / 360 * 256)
        h = h.point([(i + shift) % 256 for i in range(256)])
        out = Image.merge("HSV", (h, s, v)).convert("RGB")

    if adj["exposure"]:
        factor = 2 ** (adj["exposure"] / 50)  # ±2 stops
        out = _point_rgb(out, lambda i: i * factor)
    if adj["brightness"]:
        out = ImageEnhance.Brightness(out).enhance(1 + adj["brightness"] / 100)
    if adj["contrast"]:
        out = ImageEnhance.Contrast(out).enhance(1 + adj["contrast"] / 100)
    if adj["gamma"]:
        gamma = 2 ** (adj["gamma"] / 100)
        out = _point_rgb(out, lambda i: 255 * (i / 255) ** (1 / gamma))

    if adj["highlights"] or adj["shadows"] or adj["vibrance"] or adj["vignette"]:
        arr = np.asarray(out, dtype=np.float32) / 255.0
        if adj["highlights"] or adj["shadows"]:
            arr = _apply_highlights_shadows(arr, adj["highlights"], adj["shadows"])
        if adj["vibrance"]:
            arr = _apply_vibrance(arr, adj["vibrance"])
        if adj["vignette"]:
            arr = _apply_vignette(arr, adj["vignette"], doc_size, region, scale)
        out = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")

    if adj["saturation"]:
        out = ImageEnhance.Color(out).enhance(1 + adj["saturation"] / 100)

    if adj["temperature"] or adj["tint"]:
        r, g, b = out.split()
        if adj["temperature"]:
            warm = adj["temperature"] * 0.5
            r = r.point(_lut(lambda i: i + warm))
            b = b.point(_lut(lambda i: i - warm))
        if adj["tint"]:
            magenta = adj["tint"] * 0.5
            g = g.point(_lut(lambda i: i - magenta))
        out = Image.merge("RGB", (r, g, b))

    if adj["sepia"]:
        out = Image.blend(out, out.convert("RGB", _SEPIA_MATRIX), adj["sepia"] / 100)
    if adj["grayscale"]:
        out = Image.blend(out, out.convert("L").convert("RGB"), adj["grayscale"] / 100)

    if alpha is not None:
        return Image.merge("RGBA", (*out.split(), alpha))
    return out


def auto_enhance(img: Image.Image) -> dict[str, int]:
    """Suggest slider values that level the tonal range and neutralise a colour cast.

    Tone: the 0.5 % and 99.5 % luminance percentiles are mapped towards black
    and white via exposure + contrast.  Colour: a grey-world estimate of the
    cast becomes temperature/tint; muted pictures get a touch of vibrance.
    """
    rgb = img.convert("RGB")
    scale = min(1.0, 256 / max(rgb.size))
    if scale < 1.0:
        rgb = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    if arr.size == 0:
        return default_adjustments()
    luma = _luma(arr)
    lo, hi = float(np.percentile(luma, 0.5)), float(np.percentile(luma, 99.5))
    mean = float(luma.mean())
    adj = default_adjustments()
    spread = hi - lo
    # brightness: push the mean towards mid-grey (gently)
    adj["exposure"] = int(round(max(-40, min(40, (0.45 - mean) * 120))))
    if spread > 0.1:  # flat images (solid colours) are left alone tonally
        # contrast: stretch so the percentile range fills ~90 % of the scale
        adj["contrast"] = int(round(max(-15, min(35, (0.9 / spread - 1.0) * 40))))
        if hi < 0.85:
            adj["highlights"] = int(round(min(30, (0.85 - hi) * 120)))
        if lo > 0.12:
            adj["shadows"] = int(round(-min(30, (lo - 0.12) * 120)))
    means = arr.reshape(-1, 3).mean(axis=0)
    grey = means.mean()
    if grey > 0.02:
        red_bias = (means[0] - means[2]) / grey  # warm (+) / cool (-) cast
        green_bias = (means[1] - (means[0] + means[2]) / 2) / grey
        adj["temperature"] = int(round(max(-40, min(40, -red_bias * 60))))
        adj["tint"] = int(round(max(-30, min(30, green_bias * 60))))
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    saturation = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())
    if saturation < 0.25:
        adj["vibrance"] = int(round(min(35, (0.25 - saturation) * 160)))
    return normalize_adjustments(adj)
