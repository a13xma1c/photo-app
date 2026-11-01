"""Photoshop-style blend modes and alpha compositing on float32 NumPy arrays.

All functions take/return arrays with values in ``[0, 1]``.  Colour arrays are
shaped ``(H, W, 3)``; alpha arrays ``(H, W, 1)``.  Formulas follow the W3C
*Compositing and Blending Level 1* specification, which matches Photoshop for
the separable modes and is very close for the non-separable ones.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Array = np.ndarray
BlendFn = Callable[[Array, Array], Array]


# ------------------------------------------------------------ separable --
def _normal(cb: Array, cs: Array) -> Array:
    return cs


def _multiply(cb: Array, cs: Array) -> Array:
    return cb * cs


def _screen(cb: Array, cs: Array) -> Array:
    return cb + cs - cb * cs


def _hard_light(cb: Array, cs: Array) -> Array:
    return np.where(cs <= 0.5, cb * 2 * cs, _screen(cb, 2 * cs - 1))


def _overlay(cb: Array, cs: Array) -> Array:
    return _hard_light(cs, cb)


def _darken(cb: Array, cs: Array) -> Array:
    return np.minimum(cb, cs)


def _lighten(cb: Array, cs: Array) -> Array:
    return np.maximum(cb, cs)


def _color_dodge(cb: Array, cs: Array) -> Array:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(cs >= 1, 1.0, np.minimum(1.0, cb / np.maximum(1 - cs, 1e-9)))
    return np.where(cb <= 0, 0.0, result)


def _color_burn(cb: Array, cs: Array) -> Array:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(cs <= 0, 0.0, 1 - np.minimum(1.0, (1 - cb) / np.maximum(cs, 1e-9)))
    return np.where(cb >= 1, 1.0, result)


def _soft_light(cb: Array, cs: Array) -> Array:
    d = np.where(cb <= 0.25, ((16 * cb - 12) * cb + 4) * cb, np.sqrt(cb))
    return np.where(cs <= 0.5, cb - (1 - 2 * cs) * cb * (1 - cb), cb + (2 * cs - 1) * (d - cb))


def _difference(cb: Array, cs: Array) -> Array:
    return np.abs(cb - cs)


def _exclusion(cb: Array, cs: Array) -> Array:
    return cb + cs - 2 * cb * cs


def _linear_dodge(cb: Array, cs: Array) -> Array:
    return np.minimum(1.0, cb + cs)


def _linear_burn(cb: Array, cs: Array) -> Array:
    return np.maximum(0.0, cb + cs - 1)


def _linear_light(cb: Array, cs: Array) -> Array:
    return np.clip(cb + 2 * cs - 1, 0.0, 1.0)


def _vivid_light(cb: Array, cs: Array) -> Array:
    return np.where(cs <= 0.5, _color_burn(cb, 2 * cs), _color_dodge(cb, 2 * cs - 1))


def _pin_light(cb: Array, cs: Array) -> Array:
    return np.where(cs <= 0.5, np.minimum(cb, 2 * cs), np.maximum(cb, 2 * cs - 1))


def _subtract(cb: Array, cs: Array) -> Array:
    return np.maximum(0.0, cb - cs)


def _divide(cb: Array, cs: Array) -> Array:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cs <= 0, 1.0, np.minimum(1.0, cb / np.maximum(cs, 1e-9)))


# -------------------------------------------------------- non-separable --
def _lum(c: Array) -> Array:
    return 0.3 * c[..., 0] + 0.59 * c[..., 1] + 0.11 * c[..., 2]


def _clip_color(c: Array) -> Array:
    lum = _lum(c)[..., None]
    n = c.min(axis=-1, keepdims=True)
    x = c.max(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(n < 0, lum + (c - lum) * lum / np.maximum(lum - n, 1e-9), c)
        c = np.where(x > 1, lum + (c - lum) * (1 - lum) / np.maximum(x - lum, 1e-9), c)
    return c


def _set_lum(c: Array, lum: Array) -> Array:
    d = (lum - _lum(c))[..., None]
    return _clip_color(c + d)


def _sat(c: Array) -> Array:
    return c.max(axis=-1) - c.min(axis=-1)


def _set_sat(c: Array, s: Array) -> Array:
    mx = c.max(axis=-1, keepdims=True)
    mn = c.min(axis=-1, keepdims=True)
    rng = mx - mn
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(rng > 0, (c - mn) * s[..., None] / np.maximum(rng, 1e-9), 0.0)
    return out


def _hue(cb: Array, cs: Array) -> Array:
    return _set_lum(_set_sat(cs, _sat(cb)), _lum(cb))


def _saturation(cb: Array, cs: Array) -> Array:
    return _set_lum(_set_sat(cb, _sat(cs)), _lum(cb))


def _color(cb: Array, cs: Array) -> Array:
    return _set_lum(cs, _lum(cb))


def _luminosity(cb: Array, cs: Array) -> Array:
    return _set_lum(cb, _lum(cs))


BLEND_MODES: dict[str, BlendFn] = {
    "normal": _normal,
    "darken": _darken,
    "multiply": _multiply,
    "color burn": _color_burn,
    "linear burn": _linear_burn,
    "lighten": _lighten,
    "screen": _screen,
    "color dodge": _color_dodge,
    "linear dodge": _linear_dodge,
    "overlay": _overlay,
    "soft light": _soft_light,
    "hard light": _hard_light,
    "vivid light": _vivid_light,
    "linear light": _linear_light,
    "pin light": _pin_light,
    "difference": _difference,
    "exclusion": _exclusion,
    "subtract": _subtract,
    "divide": _divide,
    "hue": _hue,
    "saturation": _saturation,
    "color": _color,
    "luminosity": _luminosity,
}

BLEND_MODE_NAMES: tuple[str, ...] = tuple(BLEND_MODES)


def blend(mode: str, backdrop: Array, source: Array) -> Array:
    """Apply blend *mode* to colour arrays (no alpha handling)."""
    try:
        fn = BLEND_MODES[mode]
    except KeyError:
        raise ValueError(f"unknown blend mode {mode!r}") from None
    return np.clip(fn(backdrop, source), 0.0, 1.0)


def composite_over(dst_rgb: Array, dst_a: Array, src_rgb: Array, src_a: Array, mode: str = "normal") -> tuple[Array, Array]:
    """Composite a straight-alpha RGBA source *over* a straight-alpha destination.

    Implements ``Cr = (1 - ab) * Cs + ab * B(Cb, Cs)`` followed by the
    source-over operator.  Returns the new straight ``(rgb, alpha)`` arrays.
    """
    acc_p, acc_a = composite_premultiplied(dst_rgb * dst_a, dst_a, src_rgb, src_a, mode)
    return unpremultiply(acc_p, acc_a), acc_a


def composite_premultiplied(
    acc_p: Array | None, acc_a: Array | None, src_rgb: Array, src_a: Array, mode: str = "normal"
) -> tuple[Array, Array]:
    """Source-over with blend *mode* onto a **premultiplied** accumulator.

    ``acc_p``/``acc_a`` may be ``None`` for an empty backdrop.  Keeping the
    accumulator premultiplied avoids a division per layer; call
    :func:`unpremultiply` once at the end.
    """
    if acc_p is None or acc_a is None:
        return src_rgb * src_a, src_a.copy()
    if mode == "normal":
        blended = src_rgb
    else:
        backdrop = unpremultiply(acc_p, acc_a)
        blended = (1 - acc_a) * src_rgb + acc_a * blend(mode, backdrop, src_rgb)
    inv = 1 - src_a
    out_p = blended * src_a
    out_p += acc_p * inv
    out_a = src_a + acc_a * inv
    return out_p, out_a


def unpremultiply(premultiplied: Array, alpha: Array) -> Array:
    """Straight colour from premultiplied colour (0 where alpha is 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(alpha > 0, premultiplied / np.maximum(alpha, 1e-9), 0.0).astype(np.float32, copy=False)
