"""The colour pipeline: source interpretation in, display signal out.

This is the single source of truth for what the picture looks like.  The preview
and the exporter never implement colour maths of their own - they both consume a
LUT baked from :class:`ColorPipeline`, so what you see and what you render are
the same numbers by construction rather than by careful maintenance.

Chain
-----
1. signal  -> curve domain   (limited-range expansion undone for LOG curves)
2. decode transfer           -> linear, in the curve's native units
3. normalise HDR             -> linear, 1.0 = diffuse white
4. source primaries          -> output primaries
5. exposure, white balance, contrast, highlights/shadows   (linear)
6. tone map                  -> display linear
7. encode output transfer    -> display signal
8. saturation, black/white point                           (encoded)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from . import grade as grade_ops
from .camera_profiles import get_output_profile, get_profile
from .colorimetry import apply_matrix, rgb_to_rgb_matrix
from .grade import GradeParams
from .tonemap import DIFFUSE_WHITE_NITS, gamut_compress, normalise_hdr, tone_map
from .transfer import get_transfer


def signal_to_curve_domain(signal: np.ndarray, code_referenced: bool,
                           source_range: str, bit_depth: int) -> np.ndarray:
    """Undo limited-range expansion for curves defined against raw code values.

    A decoder handing us limited-range video has already mapped code 64 to 0.0
    and code 940 to 1.0.  Vendors define their LOG curves against ``code / 1023``
    instead - S-Log3 black sits at code 95, Canon and Nikon at 128, Apple at 154 -
    so those curves need the expansion put back before the formula applies.
    Getting this wrong costs roughly a stop of shadow detail and lifts black.
    """
    if not code_referenced or source_range != "tv":
        return signal
    peak = float((1 << bit_depth) - 1)
    black = 16.0 * (1 << (bit_depth - 8))
    white = 235.0 * (1 << (bit_depth - 8))
    return signal * ((white - black) / peak) + (black / peak)


@dataclass
class SourceParams:
    """How the incoming file is to be interpreted."""

    profile_id: str = "rec709"
    #: Set to override just one half of the profile; None means "use the profile".
    transfer: Optional[str] = None
    primaries: Optional[str] = None
    #: YUV matrix coefficients. Only FFmpeg uses this, but it lives here so the
    #: whole interpretation is one object.
    matrix: Optional[str] = None
    source_range: str = "tv"          # "tv" (limited) or "pc" (full)
    bit_depth: int = 10
    source_peak_nits: float = 1000.0  # only meaningful for PQ/HLG

    def resolved_transfer(self) -> str:
        return self.transfer or get_profile(self.profile_id).transfer

    def resolved_primaries(self) -> str:
        return self.primaries or get_profile(self.profile_id).primaries


@dataclass
class OutputParams:
    """Delivery space and how to get there from HDR."""

    profile_id: str = "rec709"
    tone_mapper: str = "bt2390"
    tone_mode: str = "luminance"      # "luminance" | "rgb"
    highlight_desaturation: float = 0.35
    #: How much out-of-gamut colour is folded back in rather than clipped.
    #: ``None`` means auto: on when the source gamut is wider than the target,
    #: off when they match so a same-gamut conversion stays exactly untouched.
    gamut_compression: Optional[float] = None
    target_peak_nits: float = DIFFUSE_WHITE_NITS

    def resolved_transfer(self) -> str:
        return get_output_profile(self.profile_id).transfer

    def resolved_primaries(self) -> str:
        return get_output_profile(self.profile_id).primaries


@dataclass
class PipelineParams:
    source: SourceParams = field(default_factory=SourceParams)
    output: OutputParams = field(default_factory=OutputParams)
    grade: GradeParams = field(default_factory=GradeParams)
    #: Bypass the whole chain - the A/B compare in the UI.
    bypass: bool = False

    def copy(self) -> "PipelineParams":
        return PipelineParams(
            source=replace(self.source),
            output=replace(self.output),
            grade=replace(self.grade),
            bypass=self.bypass,
        )


class ColorPipeline:
    """Evaluates :class:`PipelineParams` over RGB arrays.

    Cheap to construct - build a new one whenever the parameters change rather
    than mutating one in place, so a half-updated pipeline can never render.
    """

    def __init__(self, params: Optional[PipelineParams] = None):
        self.params = params or PipelineParams()

    # -- introspection used by the UI ------------------------------------
    @property
    def is_hdr_source(self) -> bool:
        return get_transfer(self.params.source.resolved_transfer()).kind == "hdr"

    @property
    def is_log_source(self) -> bool:
        return get_transfer(self.params.source.resolved_transfer()).kind == "log"

    def gamut_compression_amount(self) -> float:
        """Resolve the auto setting: compress only when the gamuts differ."""
        explicit = self.params.output.gamut_compression
        if explicit is not None:
            return float(explicit)
        same = (self.params.source.resolved_primaries()
                == self.params.output.resolved_primaries())
        return 0.0 if same else 1.0

    def source_peak_normalised(self) -> float:
        """Source peak in diffuse-white units, which is what the tone mapper wants."""
        tf = get_transfer(self.params.source.resolved_transfer())
        if tf.kind == "hdr":
            return max(self.params.source.source_peak_nits / DIFFUSE_WHITE_NITS, 1.0)
        if tf.kind == "log":
            # LOG holds far more range than diffuse white; the highlight headroom
            # above 100% reflectance is what the tone curve is there to recover.
            return 8.0
        return 1.0

    # -- the transform ---------------------------------------------------
    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Map source signal RGB in [0,1] to output signal RGB in [0,1]."""
        p = self.params
        signal = np.asarray(signal, dtype=np.float32)
        if p.bypass:
            return np.clip(signal, 0.0, 1.0)

        src_tf = get_transfer(p.source.resolved_transfer())
        out_tf = get_transfer(p.output.resolved_transfer())
        src_prim = p.source.resolved_primaries()
        out_prim = p.output.resolved_primaries()

        # 1-2. into the curve's own domain, then to linear.
        x = signal_to_curve_domain(signal, src_tf.code_referenced,
                                   p.source.source_range, p.source.bit_depth)
        linear = np.asarray(src_tf.decode(x), dtype=np.float32)

        # 3. HDR curves decode to their own units; put everything on diffuse white.
        linear = normalise_hdr(linear, src_tf.id,
                               source_peak_nits=p.source.source_peak_nits,
                               primaries=src_prim).astype(np.float32, copy=False)

        # 4. into the delivery gamut. Out-of-gamut colours go negative or above
        #    one here and are handled by the tone mapper's desaturation.
        if src_prim != out_prim:
            m = rgb_to_rgb_matrix(src_prim, out_prim).astype(np.float32)
            linear = apply_matrix(linear, m)

        # 5. linear-domain grade.
        g = p.grade
        linear = grade_ops.exposure(linear, g.exposure)
        linear = grade_ops.white_balance(linear, g.temperature, g.tint, out_prim)
        linear = grade_ops.contrast(linear, g.contrast)
        linear = grade_ops.highlights_shadows(linear, g.highlights, g.shadows, out_prim)

        # 6. tone map to display linear.
        display = tone_map(linear, p.output.tone_mapper,
                           source_peak=self.source_peak_normalised(),
                           mode=p.output.tone_mode,
                           desaturation=p.output.highlight_desaturation,
                           primaries=out_prim)
        display = gamut_compress(display, out_prim,
                                 self.gamut_compression_amount())
        display = np.clip(display, 0.0, 1.0)

        # 7. encode for the display.
        encoded = np.asarray(out_tf.encode(display), dtype=np.float32)

        # 8. encoded-domain trim.
        encoded = grade_ops.saturation(encoded, g.saturation, out_prim)
        encoded = grade_ops.black_white_point(encoded, g.black_point, g.white_point)
        return np.clip(encoded, 0.0, 1.0).astype(np.float32, copy=False)

    __call__ = transform
