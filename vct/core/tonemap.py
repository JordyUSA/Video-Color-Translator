"""HDR -> SDR tone mapping.

Working units
-------------
Every operator here takes linear RGB normalised so that **1.0 is SDR diffuse
white**, and returns linear display RGB in the same units.  That normalisation
is the part most naive HDR conversions get wrong: scaling by the *peak* instead
of by diffuse white drags the whole mid-tone range down and produces the
washed-out, muddy look people associate with HDR-to-SDR conversion.  ITU-R
BT.2408 fixes diffuse white at 203 cd/m2, and :func:`normalise_hdr` uses it.

Each operator is a scalar curve; :func:`tone_map` decides whether to drive it
with luminance (preserves hue, can clip a channel) or per channel (desaturates
highlights the way film does).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from .colorimetry import luminance, rgb_to_rgb_matrix, apply_matrix
from .transfer import pq_encode, pq_decode

#: ITU-R BT.2408 reference diffuse white, in cd/m2.
DIFFUSE_WHITE_NITS = 203.0

_EPS = 1e-6


# --------------------------------------------------------------------------
# HDR normalisation
# --------------------------------------------------------------------------

def hlg_system_gamma(peak_nits: float) -> float:
    """ITU-R BT.2100 HLG system gamma for a given nominal display peak."""
    return 1.2 + 0.42 * np.log10(max(peak_nits, _EPS) / 1000.0)


def hlg_ootf(scene_rgb: np.ndarray, peak_nits: float, primaries: str = "bt2020") -> np.ndarray:
    """Apply the HLG opto-optical transfer function, returning light in cd/m2."""
    gamma = hlg_system_gamma(peak_nits)
    ys = np.maximum(luminance(scene_rgb, primaries), 0.0)
    return peak_nits * np.power(ys, gamma - 1.0) * scene_rgb


def normalise_hdr(linear: np.ndarray, transfer_kind: str, *,
                  source_peak_nits: float = 1000.0,
                  primaries: str = "bt2020") -> np.ndarray:
    """Bring a decoded HDR signal into "1.0 = diffuse white" units.

    ``transfer_kind`` is the id of the source transfer function.  PQ decodes to
    an absolute fraction of 10 000 cd/m2; HLG decodes to scene light and still
    needs its OOTF.  Anything else is already relative to white.
    """
    if transfer_kind == "pq":
        return linear * 10000.0 / DIFFUSE_WHITE_NITS
    if transfer_kind == "hlg":
        return hlg_ootf(linear, source_peak_nits, primaries) / DIFFUSE_WHITE_NITS
    return linear


# --------------------------------------------------------------------------
# Scalar tone curves.  x and the return value are both in diffuse-white units.
# --------------------------------------------------------------------------

def tm_clip(x: np.ndarray, peak: float) -> np.ndarray:
    """No tone mapping - hard clip. Included as the honest 'before' reference."""
    return np.clip(x, 0.0, 1.0)


def tm_reinhard(x: np.ndarray, peak: float) -> np.ndarray:
    """Extended Reinhard with a white point, so `peak` maps exactly to 1.0.

    Reinhard et al. 2002, eq. 4.  Gentle everywhere, but flattens contrast in
    the upper mid-tones more than the filmic curves do.
    """
    w = max(peak, 1.0 + _EPS)
    return x * (1.0 + x / (w * w)) / (1.0 + x)


_HABLE = dict(A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30)


def _hable_raw(x: np.ndarray) -> np.ndarray:
    k = _HABLE
    return ((x * (k["A"] * x + k["C"] * k["B"]) + k["D"] * k["E"])
            / (x * (k["A"] * x + k["B"]) + k["D"] * k["F"])) - k["E"] / k["F"]


def tm_hable(x: np.ndarray, peak: float) -> np.ndarray:
    """Hable / "Uncharted 2" filmic curve, normalised so `peak` maps to 1.0.

    Strong shoulder and a slight toe: contrasty, and the most "graded looking"
    of the operators here.
    """
    w = max(peak, 1.0 + _EPS)
    return _hable_raw(np.maximum(x, 0.0)) / _hable_raw(np.asarray(float(w)))


def tm_mobius(x: np.ndarray, peak: float, transition: float = 0.3) -> np.ndarray:
    """Mobius (libplacebo). Linear below `transition`, hyperbolic above it.

    Leaves everything up to the transition point mathematically untouched, so
    footage that is only slightly over range comes back almost unmodified.
    """
    x = np.maximum(x, 0.0)
    peak = max(peak, 1.0 + _EPS)
    j = float(transition)
    a = -j * j * (peak - 1.0) / (j * j - 2.0 * j + peak)
    b = (j * j - 2.0 * j * peak + peak) / max(peak - 1.0, _EPS)
    scale = (b * b + 2.0 * b * j + j * j) / (b - a)
    return np.where(x <= j, x, scale * (x + a) / (x + b))


def tm_bt2390(x: np.ndarray, peak: float) -> np.ndarray:
    """ITU-R BT.2390 EETF: a Hermite roll-off applied in the PQ domain.

    The standards-body answer, and the best mid-tone preservation of the set -
    everything below the knee point is passed through untouched, so faces and
    diffuse white land exactly where they were.
    """
    x = np.maximum(x, 0.0)
    peak = max(peak, 1.0 + _EPS)
    scale = DIFFUSE_WHITE_NITS / 10000.0

    # Normalise the PQ range so source peak -> 1 and target peak (1.0) -> max_lum.
    e_src = pq_encode(x * scale)
    e_peak = float(pq_encode(np.asarray(peak * scale)))
    e_white = float(pq_encode(np.asarray(1.0 * scale)))
    if e_peak <= _EPS:
        return x
    e1 = e_src / e_peak
    max_lum = e_white / e_peak

    ks = 1.5 * max_lum - 0.5
    t = np.clip((e1 - ks) / max(1.0 - ks, _EPS), 0.0, 1.0)
    t2, t3 = t * t, t * t * t
    hermite = ((2.0 * t3 - 3.0 * t2 + 1.0) * ks
               + (t3 - 2.0 * t2 + t) * (1.0 - ks)
               + (-2.0 * t3 + 3.0 * t2) * max_lum)
    e2 = np.where(e1 < ks, e1, hermite)
    return pq_decode(np.clip(e2 * e_peak, 0.0, 1.0)) / scale


def _aces_rrt_odt_fit(v: np.ndarray) -> np.ndarray:
    """Stephen Hill's fit of the ACES RRT + Rec.709 ODT."""
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    return a / np.maximum(b, _EPS)


def tm_aces(x: np.ndarray, peak: float) -> np.ndarray:
    """ACES filmic tone curve (RRT + ODT fit), driven per channel in AP1.

    Note this is applied by :func:`tone_map` on already-matrixed RGB; the AP1
    round trip happens there so the curve sees the same primaries ACES expects.
    """
    return np.clip(_aces_rrt_odt_fit(np.maximum(x, 0.0)), 0.0, None)


def gamut_compress(rgb: np.ndarray, primaries: str = "bt709",
                   strength: float = 0.0, threshold: float = 0.8,
                   limit: float = 1.4) -> np.ndarray:
    """ACES-style gamut compression: fold out-of-gamut colour back in smoothly.

    A wide-gamut source converted to Rec.709 produces negative channel values for
    saturated colours.  Clipping those flattens them into hard-edged patches of
    pure primary and destroys the gradient between neighbouring pixels - the
    "neon" look of a bad LOG conversion.  This instead measures each pixel's
    distance from the neutral axis and compresses distances beyond `threshold`
    asymptotically toward the gamut boundary, so hue and gradient survive and
    only saturation is spent.

    Following ACES, the threshold is below 1.0, which means colours just inside
    the boundary are very slightly desaturated too.  That is the price of a
    smooth transition, and it is why `strength` defaults to off for sources that
    are already in the delivery gamut and nothing needs compressing.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb

    thr = float(np.clip(threshold, 0.0, 0.9999))
    lim = float(max(limit, 1.001))

    ac = np.max(rgb, axis=-1, keepdims=True)
    denom = np.where(np.abs(ac) > _EPS, np.abs(ac), _EPS)
    dist = np.where(np.abs(ac) > _EPS, (ac - rgb) / denom, 0.0)

    # Scale chosen so the curve reaches exactly 1.0 at `limit` and is C1 at `thr`.
    span = lim - thr
    scale = span / np.power(np.power((1.0 - thr) / span, -1.0) - 1.0, 1.0)
    nd = np.maximum(dist - thr, 0.0) / scale
    compressed = thr + scale * nd / (1.0 + nd)
    dist_out = np.where(dist < thr, dist, compressed)

    dist_out = dist + (dist_out - dist) * strength
    return (ac - dist_out * np.abs(ac)).astype(np.float32, copy=False)


@dataclass(frozen=True)
class ToneMapper:
    id: str
    label: str
    fn: Callable[[np.ndarray, float], np.ndarray]
    note: str = ""
    #: ACES is defined in its own working space and ignores the peak argument.
    uses_peak: bool = True
    working_space: str = ""


TONE_MAPPERS: Dict[str, ToneMapper] = {t.id: t for t in [
    ToneMapper("bt2390", "BT.2390 EETF", tm_bt2390,
               "ITU reference roll-off. Best mid-tone fidelity; the safe default."),
    ToneMapper("aces", "ACES filmic", tm_aces,
               "Film-like shoulder with natural highlight desaturation.",
               uses_peak=False, working_space="ap1"),
    ToneMapper("hable", "Hable (Uncharted 2)", tm_hable,
               "Contrasty filmic curve with a slight toe."),
    ToneMapper("reinhard", "Reinhard (extended)", tm_reinhard,
               "Simple and gentle; flattens upper mid-tones."),
    ToneMapper("mobius", "Mobius", tm_mobius,
               "Passes low values through untouched, rolls off only the top."),
    ToneMapper("clip", "None (clip)", tm_clip,
               "No roll-off at all. Use to see what tone mapping is doing for you.",
               uses_peak=False),
]}

TONE_MAPPER_IDS: List[str] = list(TONE_MAPPERS)


def get_tone_mapper(mapper_id: str) -> ToneMapper:
    key = str(mapper_id).strip().lower()
    if key not in TONE_MAPPERS:
        raise KeyError(f"unknown tone mapper {mapper_id!r}")
    return TONE_MAPPERS[key]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def tone_map(rgb: np.ndarray, mapper_id: str = "bt2390", *,
             source_peak: float = 1.0,
             mode: str = "luminance",
             desaturation: float = 0.0,
             primaries: str = "bt709") -> np.ndarray:
    """Tone map linear RGB given in diffuse-white units.

    Parameters
    ----------
    source_peak
        Peak of the source, in the same units (peak_nits / 203).
    mode
        ``"luminance"`` scales RGB by ``f(Y)/Y``, which preserves hue and
        saturation exactly but can leave a channel above 1.0.  ``"rgb"`` runs
        the curve per channel, which desaturates highlights the way film does.
    desaturation
        0..1.  Blends bright, out-of-gamut colours toward neutral *before* the
        curve, so saturated highlights roll off to white instead of clipping to
        a hard-edged colour patch.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    mapper = get_tone_mapper(mapper_id)
    peak = float(max(source_peak, 1.0))

    if mapper.id == "clip" and desaturation <= 0.0:
        return np.clip(rgb, 0.0, 1.0)

    if desaturation > 0.0:
        y = luminance(rgb, primaries)
        # Only affects values above diffuse white, ramping in over one stop.
        over = np.clip((np.max(rgb, axis=-1, keepdims=True) - 1.0), 0.0, None)
        w = np.clip(over / 2.0, 0.0, 1.0) * float(np.clip(desaturation, 0.0, 1.0))
        rgb = rgb * (1.0 - w) + y * w

    work = rgb
    to_work = from_work = None
    if mapper.working_space and mapper.working_space != primaries:
        to_work = rgb_to_rgb_matrix(primaries, mapper.working_space).astype(np.float32)
        from_work = rgb_to_rgb_matrix(mapper.working_space, primaries).astype(np.float32)
        work = apply_matrix(rgb, to_work)

    if mode == "rgb" or not mapper.uses_peak:
        out = mapper.fn(work, peak)
    else:
        y = np.maximum(luminance(work, primaries), 0.0)
        mapped = mapper.fn(y, peak)
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = np.where(y > _EPS, mapped / np.maximum(y, _EPS), 1.0)
        out = work * gain

    if from_work is not None:
        out = apply_matrix(out, from_work)
    return out.astype(np.float32, copy=False)
