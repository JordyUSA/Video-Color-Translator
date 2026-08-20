"""Building the FFmpeg filter chain that turns a source into gradeable RGB.

The preview and the exporter both start from this, which is the point: an
interpretation override has to mean the same thing in both, or the exported file
will not look like what was on screen.

The chain does the YUV to RGB conversion - matrix and range - and nothing else.
It deliberately does *not* let FFmpeg convert transfer characteristics or
primaries, because that is the job of the LUT baked from the colour pipeline.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..core.camera_profiles import get_output_profile, get_profile
from ..core.pipeline import SourceParams

#: Our colour space ids to the names FFmpeg's setparams/scale filters accept.
_FF_PRIMARIES = {
    "bt709": "bt709", "bt2020": "bt2020", "p3_d65": "smpte432",
    "p3_dci": "smpte431", "bt601_625": "bt470bg", "bt601_525": "smpte170m",
}
_FF_MATRIX = {
    "bt709": "bt709", "bt2020": "bt2020nc", "bt601_625": "bt470bg",
    "bt601_525": "smpte170m", "p3_d65": "bt709", "p3_dci": "bt709",
}
_FF_TRANSFER = {
    "bt709": "bt709", "srgb": "iec61966-2-1", "gamma22": "bt470m",
    "gamma26": "smpte428", "bt1886": "bt709", "linear": "linear",
    "pq": "smpte2084", "hlg": "arib-std-b67",
}


def escape_filter_path(path: str) -> str:
    """Escape a path for use inside a filtergraph argument.

    Colons separate options and backslashes escape, so a Windows-style or
    oddly-named path has to be quoted or the whole graph fails to parse.
    """
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def source_matrix_name(source: SourceParams) -> str:
    """The YUV matrix to decode with.

    An explicit override wins.  Otherwise the matrix that goes with the source
    primaries, which is right for camera footage: a camera recording Rec.2020
    primaries uses the Rec.2020 matrix.
    """
    if source.matrix:
        return _FF_MATRIX.get(source.matrix, source.matrix)
    return _FF_MATRIX.get(source.resolved_primaries(), "bt709")


def setparams_filter(source: SourceParams) -> str:
    """FFmpeg's equivalent of Premiere's "Interpret Footage".

    Relabels the stream so the decoder uses the matrix and range we intend,
    without touching the pixel data.
    """
    primaries = _FF_PRIMARIES.get(source.resolved_primaries(), "bt709")
    transfer = _FF_TRANSFER.get(source.resolved_transfer(), "bt709")
    parts = [
        f"color_primaries={primaries}",
        f"color_trc={transfer}",
        f"colorspace={source_matrix_name(source)}",
        f"range={'pc' if source.source_range == 'pc' else 'tv'}",
    ]
    return "setparams=" + ":".join(parts)


def source_to_rgb_chain(source: SourceParams, *, pix_fmt: str = "gbrp16le",
                        scale_width: Optional[int] = None,
                        scale_height: Optional[int] = None) -> List[str]:
    """Filters taking the decoded source to full-range RGB in its own encoding.

    ``in_color_matrix`` and ``in_range`` are passed to ``scale`` explicitly
    rather than relying on it picking them up from frame metadata, because that
    inheritance has changed between FFmpeg releases and a silently wrong matrix
    is a subtle, hard-to-spot colour error.
    """
    matrix = source_matrix_name(source)
    in_range = "full" if source.source_range == "pc" else "limited"

    scale_opts = [
        f"in_color_matrix={matrix}",
        f"in_range={in_range}",
        "out_range=full",
        "flags=bicubic+accurate_rnd+full_chroma_int",
    ]
    size = ""
    if scale_width:
        size = f"{scale_width}:{scale_height if scale_height else -2}:"
    elif scale_height:
        size = f"-2:{scale_height}:"

    return [
        setparams_filter(source),
        f"scale={size}" + ":".join(scale_opts),
        f"format={pix_fmt}",
    ]


def preview_chain(source: SourceParams, max_width: int) -> str:
    """Filter string for the preview decoder: RGB48 at preview resolution."""
    return ",".join(source_to_rgb_chain(source, pix_fmt="rgb48le",
                                        scale_width=max_width))


def export_chain(source: SourceParams, lut_path: str, *,
                 interp: str = "tetrahedral",
                 out_size: Optional[Tuple[int, int]] = None,
                 out_pix_fmt: Optional[str] = None) -> str:
    """Filter string for the exporter: interpret, apply the LUT, resize, retag.

    The LUT is applied before any resize so that scaling happens on the graded
    image, which is what a display would show - resizing first would blend
    colours that have not been converted yet.
    """
    filters = source_to_rgb_chain(source, pix_fmt="gbrp16le")
    filters.append(f"lut3d=file='{escape_filter_path(lut_path)}':interp={interp}")
    if out_size:
        filters.append(f"scale={out_size[0]}:{out_size[1]}:flags=lanczos+accurate_rnd")
    if out_pix_fmt:
        filters.append(f"format={out_pix_fmt}")
    return ",".join(filters)


def output_tagging_args(output_profile_id: str) -> List[str]:
    """Tag the encoded file so players do not convert it a second time."""
    profile = get_output_profile(output_profile_id)
    return [
        "-color_primaries", profile.ff_primaries,
        "-color_trc", profile.ff_transfer,
        "-colorspace", profile.ff_matrix,
        "-color_range", "tv",
    ]
