"""Primary colour adjustments.

Each operation is applied in the domain where it behaves the way a colourist
expects, which is the whole reason they are separate functions:

* exposure, white balance, contrast and highlight/shadow recovery run on
  **linear** light, before tone mapping, so they act like camera-side changes
  and are then rolled off by the tone curve rather than clipping against it;
* saturation and the black/white trim run on the **encoded** output signal,
  after tone mapping, because that is the domain they are defined in and where
  their effect is predictable.

:func:`~vct.core.pipeline.ColorPipeline` applies them in exactly that order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .colorimetry import (adaptation_matrix, apply_matrix, get_space, luminance,
                          rgb_to_xyz_matrix, xyz_to_rgb_matrix)

_EPS = 1e-6

#: Neutral reference the temperature slider adapts away from.
BASE_CCT = 6504.0
#: Slider units (-100..100) to reciprocal-megakelvin shift.
MIRED_PER_UNIT = 0.8
#: Slider units to a green/magenta shift in CIE 1960 v. At full scale this is a
#: Duv of 0.02, which is about as far off the locus as stays physically sane.
TINT_PER_UNIT = 0.0002


def _planckian_xy(t: float) -> Tuple[float, float]:
    """Kim et al. (1964) cubic approximation of the Planckian locus."""
    t2, t3 = t * t, t * t * t
    if t <= 4000.0:
        x = -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return float(x), float(y)


def _daylight_xy(t: float) -> Tuple[float, float]:
    """CIE D-series daylight locus, defined for 4000 K - 25 000 K.

    This is the locus that actually passes through D65, which is why the
    temperature slider uses it above 4000 K instead of the Planckian curve.
    """
    if t <= 7000.0:
        x = 0.244063 + 0.09911e3 / t + 2.9678e6 / t**2 - 4.6070e9 / t**3
    else:
        x = 0.237040 + 0.24748e3 / t + 1.9018e6 / t**2 - 2.0064e9 / t**3
    y = -3.000 * x * x + 2.870 * x - 0.275
    return float(x), float(y)


# The two loci disagree by a small Duv at their 4000 K meeting point.  Offsetting
# the Planckian half by that difference keeps the slider continuous, so nudging
# temperature off zero can never snap the tint.
_LOCUS_JOIN_K = 4000.0
_LOCUS_OFFSET = tuple(
    d - p for d, p in zip(_daylight_xy(_LOCUS_JOIN_K), _planckian_xy(_LOCUS_JOIN_K))
)


def cct_to_xy(cct: float) -> Tuple[float, float]:
    """Chromaticity for a correlated colour temperature, 1667 K - 25 000 K.

    Daylight locus above 4000 K (so 6504 K is exactly D65), Planckian locus
    below it for tungsten and candlelight, joined continuously.
    """
    t = float(np.clip(cct, 1667.0, 25000.0))
    if t >= _LOCUS_JOIN_K:
        return _daylight_xy(t)
    x, y = _planckian_xy(t)
    return x + _LOCUS_OFFSET[0], y + _LOCUS_OFFSET[1]


def _xy_to_uv(xy: Tuple[float, float]) -> Tuple[float, float]:
    """CIE 1960 UCS, the space the tint offset is perpendicular in."""
    x, y = xy
    d = -2.0 * x + 12.0 * y + 3.0
    return 4.0 * x / d, 6.0 * y / d


def _uv_to_xy(uv: Tuple[float, float]) -> Tuple[float, float]:
    u, v = uv
    d = 2.0 * u - 8.0 * v + 4.0
    return 3.0 * u / d, 2.0 * v / d


def _locus_normal(cct: float) -> Tuple[float, float]:
    """Unit vector perpendicular to the locus in CIE 1960 UCS at `cct`.

    This is the Duv direction - the actual green/magenta axis.  Offsetting along
    a fixed axis instead (say, v alone) would drag the temperature along with the
    tint, because the locus is not parallel to either axis.
    """
    d = max(cct * 0.01, 1.0)
    u0, v0 = _xy_to_uv(cct_to_xy(cct - d))
    u1, v1 = _xy_to_uv(cct_to_xy(cct + d))
    tu, tv = u1 - u0, v1 - v0
    n = (tu * tu + tv * tv) ** 0.5
    if n < _EPS:
        return 0.0, 1.0
    # Rotate the tangent 90 degrees. The sign assumes a greener source white for
    # positive tint, so the corrected image moves toward magenta - matching the
    # direction the slider is labelled in Premiere and Lightroom.
    return tv / n, -tu / n


def white_balance_white_point(temperature: float, tint: float) -> Tuple[float, float]:
    """The white point the image is assumed to have been shot under.

    Positive ``temperature`` assumes a bluer (higher-CCT) source, which makes the
    corrected image warmer - the direction the slider label promises.  Positive
    ``tint`` assumes a greener source and pushes the result toward magenta.
    """
    if abs(temperature) < _EPS and abs(tint) < _EPS:
        return cct_to_xy(BASE_CCT)
    mired = 1.0e6 / BASE_CCT - float(temperature) * MIRED_PER_UNIT
    cct = 1.0e6 / max(mired, 1.0e6 / 25000.0)
    u, v = _xy_to_uv(cct_to_xy(cct))
    nu, nv = _locus_normal(cct)
    u += nu * float(tint) * TINT_PER_UNIT
    v += nv * float(tint) * TINT_PER_UNIT
    x, y = _uv_to_xy((u, v))
    # Guard the extremes: an xy outside the visible triangle makes the adaptation
    # matrix blow up and turns a slider into a broken image.
    x = float(np.clip(x, 0.05, 0.85))
    y = float(np.clip(y, 0.05, 0.85))
    if x + y > 0.98:
        scale = 0.98 / (x + y)
        x, y = x * scale, y * scale
    return x, y


# --------------------------------------------------------------------------
# Linear-domain operations
# --------------------------------------------------------------------------

def exposure(rgb: np.ndarray, stops: float) -> np.ndarray:
    """Scale linear light by 2**stops - exactly what opening the iris does."""
    if abs(stops) < _EPS:
        return rgb
    return rgb * np.float32(2.0 ** float(stops))


def white_balance_matrix(temperature: float, tint: float, primaries: str,
                         cat: str = "cat02") -> np.ndarray:
    """3x3 white balance matrix for linear RGB in `primaries`.

    A real chromatic adaptation rather than a per-channel gain, so neutrals stay
    neutral and skin tones do not swing magenta as the slider moves.
    """
    space = get_space(primaries)
    src_white = white_balance_white_point(temperature, tint)
    m = adaptation_matrix(src_white, space.white, cat)
    return xyz_to_rgb_matrix(space) @ m @ rgb_to_xyz_matrix(space)


def white_balance(rgb: np.ndarray, temperature: float, tint: float,
                  primaries: str = "bt709") -> np.ndarray:
    if abs(temperature) < _EPS and abs(tint) < _EPS:
        return rgb
    m = white_balance_matrix(temperature, tint, primaries).astype(np.float32)
    return apply_matrix(rgb, m)


def contrast(rgb: np.ndarray, amount: float, pivot: float = 0.18) -> np.ndarray:
    """Pivoted contrast: a straight-line slope change in log exposure space.

    ``amount`` is in stops of slope, so 0 is neutral, +1 doubles the log slope
    and -1 halves it.  Mid grey stays exactly where it is.
    """
    if abs(amount) < _EPS:
        return rgb
    k = np.float32(2.0 ** float(amount))
    p = np.float32(pivot)
    return p * np.power(np.maximum(rgb, 0.0) / p, k)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def highlights_shadows(rgb: np.ndarray, highlights: float, shadows: float,
                       primaries: str = "bt709") -> np.ndarray:
    """Luminance-masked recovery, in stops, applied as a gain on linear light.

    Because it is a gain rather than an offset, pushing shadows up brightens
    what is there without lifting true black into a milky haze.
    """
    if abs(highlights) < _EPS and abs(shadows) < _EPS:
        return rgb
    y = np.maximum(luminance(rgb, primaries), 0.0)
    # Perceptual position of each pixel, 0 at black and 1 near diffuse white.
    n = np.power(np.clip(y, 0.0, 1.0), np.float32(1.0 / 2.4))
    shadow_mask = _smoothstep((0.5 - n) / 0.5)
    highlight_mask = _smoothstep((n - 0.5) / 0.5)
    gain = np.exp2(np.float32(shadows) * shadow_mask
                   + np.float32(highlights) * highlight_mask)
    return rgb * gain


# --------------------------------------------------------------------------
# Encoded-domain operations
# --------------------------------------------------------------------------

def saturation(rgb: np.ndarray, amount: float, primaries: str = "bt709") -> np.ndarray:
    """Scale distance from the neutral axis. 0 is neutral, -1 is monochrome."""
    if abs(amount) < _EPS:
        return rgb
    s = np.float32(1.0 + float(amount))
    y = luminance(rgb, primaries)
    return y + (rgb - y) * s


def black_white_point(rgb: np.ndarray, black: float, white: float) -> np.ndarray:
    """Linear trim of the encoded signal: black lands at 0, white lands at 1."""
    if abs(black) < _EPS and abs(white - 1.0) < _EPS:
        return rgb
    span = max(float(white) - float(black), 1e-3)
    return (rgb - np.float32(black)) / np.float32(span)


# --------------------------------------------------------------------------
# Parameter block
# --------------------------------------------------------------------------

@dataclass
class GradeParams:
    """Everything the adjustment panel controls. Defaults are all no-ops."""

    exposure: float = 0.0        # stops
    contrast: float = 0.0        # stops of log slope
    saturation: float = 0.0      # -1 monochrome .. +1 double
    temperature: float = 0.0     # -100 cooler .. +100 warmer
    tint: float = 0.0            # -100 green .. +100 magenta
    highlights: float = 0.0      # stops
    shadows: float = 0.0         # stops
    black_point: float = 0.0
    white_point: float = 1.0

    def is_neutral(self) -> bool:
        return (abs(self.exposure) < _EPS and abs(self.contrast) < _EPS
                and abs(self.saturation) < _EPS and abs(self.temperature) < _EPS
                and abs(self.tint) < _EPS and abs(self.highlights) < _EPS
                and abs(self.shadows) < _EPS and abs(self.black_point) < _EPS
                and abs(self.white_point - 1.0) < _EPS)

    def reset(self) -> None:
        for f, v in GradeParams().__dict__.items():
            setattr(self, f, v)
