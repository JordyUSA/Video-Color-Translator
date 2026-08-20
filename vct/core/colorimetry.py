"""RGB colour space definitions and the matrices that convert between them.

Everything here is derived from CIE xy chromaticity coordinates rather than
hard-coded matrices, so a new camera gamut only needs its published primaries.

References:
    ITU-R BT.709-6, ITU-R BT.2020-2, SMPTE RP 431-2 (DCI-P3),
    Sony "S-Gamut3/S-Gamut3.cine Technical Summary",
    Panasonic "V-Log/V-Gamut Reference Manual",
    Canon "Cinema Gamut / Canon Log White Paper",
    ARRI "ALEXA LogC Curve Usage in VFX",
    Blackmagic Design "Blackmagic Generation 5 Color Science",
    Academy TB-2014-004 (ACES AP0/AP1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# White points (CIE 1931 xy)
# --------------------------------------------------------------------------

WHITE_POINTS: Dict[str, Tuple[float, float]] = {
    "D65": (0.3127, 0.3290),
    "D60": (0.32168, 0.33767),   # ACES white
    "D55": (0.33242, 0.34743),
    "D50": (0.34567, 0.35850),
    "DCI": (0.31400, 0.35100),   # DCI calibration white (~6300K green-ish)
    "E": (1.0 / 3.0, 1.0 / 3.0),
}


@dataclass(frozen=True)
class ColorSpace:
    """An RGB colour space defined by its primaries and white point."""

    id: str
    label: str
    primaries: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    white: Tuple[float, float]
    note: str = ""


def _p(r, g, b):
    return (r, g, b)


COLOR_SPACES: Dict[str, ColorSpace] = {
    cs.id: cs
    for cs in [
        ColorSpace(
            "bt709", "Rec.709 / sRGB",
            _p((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)), WHITE_POINTS["D65"],
            "HD broadcast and sRGB share these primaries.",
        ),
        ColorSpace(
            "bt2020", "Rec.2020",
            _p((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)), WHITE_POINTS["D65"],
            "UHD / HDR container gamut.",
        ),
        ColorSpace(
            "bt601_625", "Rec.601 (625/PAL)",
            _p((0.640, 0.330), (0.290, 0.600), (0.150, 0.060)), WHITE_POINTS["D65"],
        ),
        ColorSpace(
            "bt601_525", "Rec.601 (525/NTSC)",
            _p((0.630, 0.340), (0.310, 0.595), (0.155, 0.070)), WHITE_POINTS["D65"],
        ),
        ColorSpace(
            "p3_d65", "Display P3",
            _p((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)), WHITE_POINTS["D65"],
            "Apple displays; DCI primaries with a D65 white point.",
        ),
        ColorSpace(
            "p3_dci", "DCI-P3 (theatrical)",
            _p((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)), WHITE_POINTS["DCI"],
        ),
        ColorSpace(
            "s_gamut3", "Sony S-Gamut3",
            _p((0.730, 0.280), (0.140, 0.855), (0.100, -0.050)), WHITE_POINTS["D65"],
        ),
        ColorSpace(
            "s_gamut3_cine", "Sony S-Gamut3.cine",
            _p((0.766, 0.275), (0.225, 0.800), (0.089, -0.087)), WHITE_POINTS["D65"],
            "Easier to grade than S-Gamut3; the usual choice for S-Log3.",
        ),
        ColorSpace(
            "v_gamut", "Panasonic V-Gamut",
            _p((0.730, 0.280), (0.165, 0.840), (0.100, -0.030)), WHITE_POINTS["D65"],
        ),
        ColorSpace(
            "cinema_gamut", "Canon Cinema Gamut",
            _p((0.740, 0.270), (0.170, 1.140), (0.080, -0.100)), WHITE_POINTS["D65"],
        ),
        ColorSpace(
            "awg3", "ARRI Wide Gamut 3",
            _p((0.6840, 0.3130), (0.2210, 0.8480), (0.0861, -0.1020)), WHITE_POINTS["D65"],
        ),
        ColorSpace(
            "bmdwg", "Blackmagic Wide Gamut",
            _p((0.7177215, 0.3171181), (0.2280410, 0.8615690), (0.1005841, -0.0820452)),
            (0.3127170, 0.3290312),
        ),
        ColorSpace(
            "ap0", "ACES 2065-1 (AP0)",
            _p((0.7347, 0.2653), (0.0000, 1.0000), (0.0001, -0.0770)), WHITE_POINTS["D60"],
        ),
        ColorSpace(
            "ap1", "ACEScg (AP1)",
            _p((0.713, 0.293), (0.165, 0.830), (0.128, 0.044)), WHITE_POINTS["D60"],
        ),
    ]
}

# Aliases so callers can use the names FFmpeg/containers use.
COLOR_SPACE_ALIASES: Dict[str, str] = {
    "srgb": "bt709",
    "rec709": "bt709",
    "bt470bg": "bt601_625",
    "smpte170m": "bt601_525",
    "smpte240m": "bt601_525",
    "bt2020_ncl": "bt2020",
    "bt2020_cl": "bt2020",
    "smpte431": "p3_dci",
    "smpte432": "p3_d65",
    "display_p3": "p3_d65",
}


def get_space(space_id: str) -> ColorSpace:
    """Look up a colour space by id or alias. Raises KeyError with a useful message."""
    key = space_id.strip().lower()
    key = COLOR_SPACE_ALIASES.get(key, key)
    if key not in COLOR_SPACES:
        raise KeyError(f"unknown colour space {space_id!r}")
    return COLOR_SPACES[key]


# --------------------------------------------------------------------------
# Chromaticity -> matrix
# --------------------------------------------------------------------------

def xy_to_XYZ(xy: Sequence[float], Y: float = 1.0) -> np.ndarray:
    """Convert a CIE xy chromaticity to XYZ at the given luminance."""
    x, y = float(xy[0]), float(xy[1])
    if y == 0.0:
        return np.zeros(3, dtype=np.float64)
    return np.array([x * Y / y, Y, (1.0 - x - y) * Y / y], dtype=np.float64)


def rgb_to_xyz_matrix(space: ColorSpace) -> np.ndarray:
    """Build the 3x3 RGB->XYZ matrix for a colour space (SMPTE RP 177 method)."""
    m = np.column_stack([xy_to_XYZ(p) for p in space.primaries])
    white = xy_to_XYZ(space.white)
    scale = np.linalg.solve(m, white)
    return m * scale  # column-wise scaling


def xyz_to_rgb_matrix(space: ColorSpace) -> np.ndarray:
    return np.linalg.inv(rgb_to_xyz_matrix(space))


# Chromatic adaptation transforms.
CAT_MATRICES: Dict[str, np.ndarray] = {
    "bradford": np.array([
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ]),
    "cat02": np.array([
        [0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975, 0.0061],
        [0.0030, 0.0136, 0.9834],
    ]),
    "vonkries": np.array([
        [0.40024, 0.70760, -0.08081],
        [-0.22630, 1.16532, 0.04570],
        [0.0, 0.0, 0.91822],
    ]),
    "xyzscaling": np.eye(3),
}


def adaptation_matrix(src_white: Sequence[float], dst_white: Sequence[float],
                      method: str = "bradford") -> np.ndarray:
    """Von Kries style chromatic adaptation between two xy white points."""
    if tuple(src_white) == tuple(dst_white):
        return np.eye(3)
    m = CAT_MATRICES[method]
    src = m @ xy_to_XYZ(src_white)
    dst = m @ xy_to_XYZ(dst_white)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.diag(np.where(src != 0, dst / src, 1.0))
    return np.linalg.inv(m) @ gain @ m


def rgb_to_rgb_matrix(src: str | ColorSpace, dst: str | ColorSpace,
                      cat: str = "bradford") -> np.ndarray:
    """3x3 matrix converting linear RGB in `src` primaries to linear RGB in `dst`.

    White point differences are handled by chromatic adaptation, so e.g.
    DCI-P3 -> Rec.709 keeps neutrals neutral instead of shifting them green.
    """
    s = get_space(src) if isinstance(src, str) else src
    d = get_space(dst) if isinstance(dst, str) else dst
    if s.id == d.id:
        return np.eye(3)
    m = xyz_to_rgb_matrix(d) @ adaptation_matrix(s.white, d.white, cat) @ rgb_to_xyz_matrix(s)
    return m


def apply_matrix(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix to an array whose last axis is RGB."""
    if matrix.shape != (3, 3):
        raise ValueError("matrix must be 3x3")
    return rgb @ matrix.T.astype(rgb.dtype, copy=False)


# --------------------------------------------------------------------------
# Luminance
# --------------------------------------------------------------------------

def luma_weights(space: str | ColorSpace = "bt709") -> np.ndarray:
    """Y row of the RGB->XYZ matrix: the true luminance weights for the space."""
    s = get_space(space) if isinstance(space, str) else space
    return rgb_to_xyz_matrix(s)[1].astype(np.float64)


def luminance(rgb: np.ndarray, space: str | ColorSpace = "bt709") -> np.ndarray:
    """Relative luminance of linear RGB, keeping the trailing axis for broadcasting."""
    w = luma_weights(space).astype(rgb.dtype, copy=False)
    return np.sum(rgb * w, axis=-1, keepdims=True)
