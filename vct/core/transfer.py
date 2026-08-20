"""Transfer functions: every OETF/EOTF this tool can interpret or output.

Conventions
-----------
``encode(linear) -> signal`` and ``decode(signal) -> linear`` are exact inverses
over the curve's working domain.  What "linear" means depends on the curve:

* SDR display curves (BT.709, sRGB, gamma 2.2/2.4) - display light, 1.0 = white.
* Camera LOG curves - scene-referred reflectance, 0.18 = mid grey, 1.0 = 100%.
* PQ - absolute display light as a fraction of 10 000 cd/m2 (BT.2100).
* HLG - scene light in [0, 1] per the reference OETF (no OOTF applied here).

The pipeline, not this module, converts those different "linear" meanings into
the single normalised space it grades in.  Keeping that out of here is what lets
every curve be tested as a pure round trip.

``code_referenced`` marks curves whose published formula is defined against raw
code values (code / (2**n - 1)) rather than against the full-scale signal you get
after limited-range expansion.  Every camera LOG curve is in this group: Sony
puts S-Log3 black at 10-bit code 95, Canon and Nikon put theirs at 128, Apple at
154.  A decoder handing us limited-range video maps code 64 to 0.0, so those
values have to be mapped back into code space before the curve is applied - see
``pipeline.source_signal_to_curve_domain``.

References
----------
ITU-R BT.709-6, ITU-R BT.2100-2 (PQ and HLG), IEC 61966-2-1 (sRGB),
Sony "S-Log3 Technical Summary" (v1.12), Panasonic "V-Log/V-Gamut Reference
Manual" (rev 1.0), Canon "Canon Log Gamma Curves White Paper" (2018),
Nikon "N-Log Specification Document" (v1.0.0), ARRI "ALEXA LogC Curve
Usage in VFX" (2017), Apple "Apple Log Profile White Paper" (Sept 2023),
Blackmagic Design "Blackmagic Generation 5 Color Science" (2020).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

ArrayLike = np.ndarray

_TINY = 1e-10


def _asarray(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind != "f":
        a = a.astype(np.float64)
    return a


def _log10(x: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(x, _TINY))


def _log2(x: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(x, _TINY))


def _spow(x: np.ndarray, p: float) -> np.ndarray:
    """Sign-preserving power, so negative excursions survive instead of becoming NaN."""
    return np.sign(x) * np.abs(x) ** p


# --------------------------------------------------------------------------
# Standard display curves
# --------------------------------------------------------------------------

def bt709_encode(L):
    L = _asarray(L)
    return np.where(L < 0.018, 4.5 * L, 1.099 * _spow(L, 0.45) - 0.099)


def bt709_decode(V):
    V = _asarray(V)
    return np.where(V < 0.081, V / 4.5, _spow((V + 0.099) / 1.099, 1.0 / 0.45))


def srgb_encode(L):
    L = _asarray(L)
    return np.where(L <= 0.0031308, L * 12.92, 1.055 * _spow(L, 1.0 / 2.4) - 0.055)


def srgb_decode(V):
    V = _asarray(V)
    return np.where(V <= 0.04045, V / 12.92, _spow((V + 0.055) / 1.055, 2.4))


def _gamma_pair(g: float):
    return (lambda L, g=g: _spow(_asarray(L), 1.0 / g),
            lambda V, g=g: _spow(_asarray(V), g))


def linear_encode(L):
    return _asarray(L)


def linear_decode(V):
    return _asarray(V)


# --------------------------------------------------------------------------
# HDR curves
# --------------------------------------------------------------------------

# SMPTE ST 2084 / ITU-R BT.2100 PQ
_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 4096.0 * 128.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 4096.0 * 32.0
_PQ_C3 = 2392.0 / 4096.0 * 32.0


def pq_encode(L):
    """Absolute luminance as a fraction of 10 000 cd/m2 -> PQ signal."""
    Lm = np.power(np.clip(_asarray(L), 0.0, None), _PQ_M1)
    return np.power((_PQ_C1 + _PQ_C2 * Lm) / (1.0 + _PQ_C3 * Lm), _PQ_M2)


def pq_decode(V):
    """PQ signal -> absolute luminance as a fraction of 10 000 cd/m2."""
    Vm = np.power(np.clip(_asarray(V), 0.0, None), 1.0 / _PQ_M2)
    num = np.maximum(Vm - _PQ_C1, 0.0)
    den = np.maximum(_PQ_C2 - _PQ_C3 * Vm, _TINY)
    return np.power(num / den, 1.0 / _PQ_M1)


# ARIB STD-B67 / ITU-R BT.2100 HLG
_HLG_A = 0.17883277
_HLG_B = 1.0 - 4.0 * _HLG_A          # 0.28466892
_HLG_C = 0.5 - _HLG_A * np.log(4.0 * _HLG_A)   # 0.55991073


def hlg_encode(E):
    E = np.clip(_asarray(E), 0.0, None)
    return np.where(E <= 1.0 / 12.0,
                    np.sqrt(3.0 * E),
                    _HLG_A * np.log(np.maximum(12.0 * E - _HLG_B, _TINY)) + _HLG_C)


def hlg_decode(V):
    V = np.clip(_asarray(V), 0.0, None)
    return np.where(V <= 0.5,
                    V * V / 3.0,
                    (np.exp((V - _HLG_C) / _HLG_A) + _HLG_B) / 12.0)


# --------------------------------------------------------------------------
# Camera LOG curves
# --------------------------------------------------------------------------

def slog3_encode(x):
    """Sony S-Log3. 18% grey -> 10-bit code 420 (0.41056)."""
    x = _asarray(x)
    return np.where(
        x >= 0.01125000,
        (420.0 + _log10((x + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0,
        (x * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0,
    )


def slog3_decode(y):
    y = _asarray(y)
    return np.where(
        y >= 171.2102946929 / 1023.0,
        np.power(10.0, (y * 1023.0 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01,
        (y * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0),
    )


_VLOG_CUT1, _VLOG_CUT2 = 0.01, 0.181
_VLOG_B, _VLOG_C, _VLOG_D = 0.00873, 0.241514, 0.598206


def vlog_encode(x):
    """Panasonic V-Log. 18% grey -> 0.42335 (10-bit code 433)."""
    x = _asarray(x)
    return np.where(x < _VLOG_CUT1,
                    5.6 * x + 0.125,
                    _VLOG_C * _log10(x + _VLOG_B) + _VLOG_D)


def vlog_decode(y):
    y = _asarray(y)
    return np.where(y < _VLOG_CUT2,
                    (y - 0.125) / 5.6,
                    np.power(10.0, (y - _VLOG_D) / _VLOG_C) - _VLOG_B)


_LOGC3 = dict(cut=0.010591, a=5.555556, b=0.052272, c=0.247190,
              d=0.385537, e=5.367655, f=0.092809)


def logc3_encode(x):
    """ARRI LogC3 at EI 800. 18% grey -> 0.391006."""
    x = _asarray(x)
    k = _LOGC3
    return np.where(x > k["cut"],
                    k["c"] * _log10(k["a"] * x + k["b"]) + k["d"],
                    k["e"] * x + k["f"])


def logc3_decode(y):
    y = _asarray(y)
    k = _LOGC3
    return np.where(y > k["e"] * k["cut"] + k["f"],
                    (np.power(10.0, (y - k["d"]) / k["c"]) - k["b"]) / k["a"],
                    (y - k["f"]) / k["e"])


def _legal_pack(ire: np.ndarray, bits: int = 10) -> np.ndarray:
    """Canon publishes its curves in IRE; the recorded code value is legal-range."""
    peak = float((1 << bits) - 1)
    black = 16.0 * (1 << (bits - 8))
    white = 235.0 * (1 << (bits - 8))
    return (ire * (white - black) + black) / peak


def _legal_unpack(code: np.ndarray, bits: int = 10) -> np.ndarray:
    peak = float((1 << bits) - 1)
    black = 16.0 * (1 << (bits - 8))
    white = 235.0 * (1 << (bits - 8))
    return (code * peak - black) / (white - black)


def clog2_encode(x):
    """Canon Log 2. 18% grey -> 0.40355 code (39.82 IRE)."""
    x = _asarray(x) / 0.9
    ire = np.where(x < -0.02,
                   -0.24136077 * _log10(-x * 87.09937 + 1.0) + 0.092864125,
                   0.24136077 * _log10(x * 87.09937 + 1.0) + 0.092864125)
    return _legal_pack(ire)


def clog2_decode(y):
    ire = _legal_unpack(_asarray(y))
    x = np.where(ire < 0.092864125,
                 -(np.power(10.0, (0.092864125 - ire) / 0.24136077) - 1.0) / 87.09937,
                 (np.power(10.0, (ire - 0.092864125) / 0.24136077) - 1.0) / 87.09937)
    return x * 0.9


def clog3_encode(x):
    """Canon Log 3. 18% grey -> 0.34339 code (32.80 IRE)."""
    x = _asarray(x) / 0.9
    ire = np.where(
        x < -0.014,
        -0.42889912 * _log10(-x * 14.98325 + 1.0) + 0.07623209,
        np.where(x <= 0.014,
                 2.3271410 * x + 0.073059361,
                 0.42889912 * _log10(x * 14.98325 + 1.0) + 0.069886632),
    )
    return _legal_pack(ire)


def clog3_decode(y):
    ire = _legal_unpack(_asarray(y))
    lo = 2.3271410 * -0.014 + 0.073059361
    hi = 2.3271410 * 0.014 + 0.073059361
    x = np.where(
        ire < lo,
        -(np.power(10.0, (0.07623209 - ire) / 0.42889912) - 1.0) / 14.98325,
        np.where(ire <= hi,
                 (ire - 0.073059361) / 2.3271410,
                 (np.power(10.0, (ire - 0.069886632) / 0.42889912) - 1.0) / 14.98325),
    )
    return x * 0.9


#: Nikon's two published branches do not meet exactly at the cut - they differ by
#: about 1.6e-4. Deriving the decode threshold from the encode's own crossover
#: instead of the rounded code 452 keeps the pair an exact inverse.
_NLOG_CUT_X = 0.328
_NLOG_CUT_Y = 150.0 / 1023.0 * np.log(_NLOG_CUT_X) + 619.0 / 1023.0


def nlog_encode(x):
    """Nikon N-Log. 18% grey -> 0.36364 (10-bit code 372)."""
    x = _asarray(x)
    return np.where(x > _NLOG_CUT_X,
                    150.0 / 1023.0 * np.log(np.maximum(x, _TINY)) + 619.0 / 1023.0,
                    650.0 / 1023.0 * np.cbrt(np.maximum(x + 0.0075, 0.0)))


def nlog_decode(y):
    y = _asarray(y)
    return np.where(y >= _NLOG_CUT_Y,
                    np.exp((y * 1023.0 - 619.0) / 150.0),
                    np.power(y * 1023.0 / 650.0, 3.0) - 0.0075)


_APPLE = dict(R0=-0.05641088, Rt=0.01, c=47.28711236,
              beta=0.00964052, gamma=0.08550479, delta=0.69336945)
_APPLE_PT = _APPLE["c"] * (_APPLE["Rt"] - _APPLE["R0"]) ** 2


def applelog_encode(x):
    """Apple Log. 18% grey -> 0.48876 (10-bit code 500), 90% white -> code 697."""
    x = _asarray(x)
    k = _APPLE
    return np.where(
        x < k["R0"], 0.0,
        np.where(x < k["Rt"],
                 k["c"] * (x - k["R0"]) ** 2,
                 k["gamma"] * _log2(x + k["beta"]) + k["delta"]),
    )


def applelog_decode(y):
    y = _asarray(y)
    k = _APPLE
    return np.where(
        y < 0.0, k["R0"],
        np.where(y < _APPLE_PT,
                 np.sqrt(np.maximum(y, 0.0) / k["c"]) + k["R0"],
                 np.power(2.0, (y - k["delta"]) / k["gamma"]) - k["beta"]),
    )


_BMD = dict(A=0.08692876065491224, B=0.005494072432257808,
            C=0.5300133392291939, D=8.283605932402494,
            E=0.09246575342465753, cut=0.005)


def bmdfilm5_encode(x):
    """Blackmagic Film Generation 5. 18% grey -> 0.38355."""
    x = _asarray(x)
    k = _BMD
    return np.where(x < k["cut"],
                    k["D"] * x + k["E"],
                    k["A"] * np.log(np.maximum(x + k["B"], _TINY)) + k["C"])


def bmdfilm5_decode(y):
    y = _asarray(y)
    k = _BMD
    return np.where(y < k["D"] * k["cut"] + k["E"],
                    (y - k["E"]) / k["D"],
                    np.exp((y - k["C"]) / k["A"]) - k["B"])


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TransferFunction:
    id: str
    label: str
    encode: Callable[[ArrayLike], np.ndarray]
    decode: Callable[[ArrayLike], np.ndarray]
    kind: str                      # "sdr" | "hdr" | "log" | "linear"
    code_referenced: bool = False  # see module docstring
    grey: Optional[float] = None   # encoded value of 0.18 scene linear
    note: str = ""


def _tf(*args, **kwargs) -> TransferFunction:
    return TransferFunction(*args, **kwargs)


_g22e, _g22d = _gamma_pair(2.2)
_g24e, _g24d = _gamma_pair(2.4)
_g26e, _g26d = _gamma_pair(2.6)

TRANSFERS: Dict[str, TransferFunction] = {t.id: t for t in [
    _tf("bt709", "Rec.709 (OETF)", bt709_encode, bt709_decode, "sdr",
        note="Broadcast camera curve. Displays usually apply BT.1886 (gamma 2.4)."),
    _tf("srgb", "sRGB", srgb_encode, srgb_decode, "sdr",
        note="Computer display standard; ~gamma 2.2 with a linear toe."),
    _tf("gamma22", "Gamma 2.2", _g22e, _g22d, "sdr"),
    _tf("bt1886", "BT.1886 / Gamma 2.4", _g24e, _g24d, "sdr",
        note="Reference display EOTF for Rec.709 grading."),
    _tf("gamma26", "Gamma 2.6 (DCI)", _g26e, _g26d, "sdr"),
    _tf("linear", "Linear", linear_encode, linear_decode, "linear"),
    _tf("pq", "PQ / SMPTE ST 2084", pq_encode, pq_decode, "hdr",
        note="Absolute HDR. 1.0 = 10 000 cd/m2; diffuse white sits near 203 cd/m2."),
    _tf("hlg", "HLG / ARIB STD-B67", hlg_encode, hlg_decode, "hdr",
        note="Relative HDR. Reference white at 75% signal; OOTF applied by the display."),
    _tf("slog3", "Sony S-Log3", slog3_encode, slog3_decode, "log",
        code_referenced=True, grey=420.0 / 1023.0,
        note="Sony FX/A7S/Venice. Black at 10-bit code 95."),
    _tf("vlog", "Panasonic V-Log", vlog_encode, vlog_decode, "log",
        code_referenced=True, grey=0.42335,
        note="Panasonic S/GH/Varicam. Black at code 128."),
    _tf("logc3", "ARRI LogC3 (EI 800)", logc3_encode, logc3_decode, "log",
        code_referenced=True, grey=0.391006,
        note="ALEXA Classic/Mini/LF. This is the EI 800 curve."),
    _tf("clog2", "Canon Log 2", clog2_encode, clog2_decode, "log",
        code_referenced=True, grey=0.40355,
        note="Widest Canon curve; needs a grade. Black at code 128."),
    _tf("clog3", "Canon Log 3", clog3_encode, clog3_decode, "log",
        code_referenced=True, grey=0.34339,
        note="Canon's easier-to-grade curve. Black at code 128."),
    _tf("nlog", "Nikon N-Log", nlog_encode, nlog_decode, "log",
        code_referenced=True, grey=0.363636,
        note="Nikon Z series. Black at code 128."),
    _tf("applelog", "Apple Log", applelog_encode, applelog_decode, "log",
        code_referenced=True, grey=0.488757,
        note="iPhone 15 Pro and later. 18% grey at code 500, 90% white at 697."),
    _tf("bmdfilm5", "Blackmagic Film Gen 5", bmdfilm5_encode, bmdfilm5_decode, "log",
        code_referenced=True, grey=0.383547),
]}

# Names as they appear in FFmpeg / container metadata.
TRANSFER_ALIASES: Dict[str, str] = {
    "bt470m": "gamma22",
    "bt470bg": "gamma22",
    "smpte170m": "bt709",
    "smpte240m": "bt709",
    "bt1361e": "bt709",
    "bt2020-10": "bt709",
    "bt2020_10": "bt709",
    "bt2020-12": "bt709",
    "bt2020_12": "bt709",
    "iec61966-2-1": "srgb",
    "iec61966_2_1": "srgb",
    "iec61966-2-4": "bt709",
    "smpte2084": "pq",
    "smpte428": "gamma26",
    "arib-std-b67": "hlg",
    "arib_std_b67": "hlg",
    "log": "linear",
    "log100": "linear",
    "log316": "linear",
}


def get_transfer(transfer_id: str) -> TransferFunction:
    key = str(transfer_id).strip().lower().replace(" ", "")
    key = TRANSFER_ALIASES.get(key, key)
    if key not in TRANSFERS:
        raise KeyError(f"unknown transfer function {transfer_id!r}")
    return TRANSFERS[key]
