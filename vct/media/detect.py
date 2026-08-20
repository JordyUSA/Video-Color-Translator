"""Suggesting how a file should be interpreted, and saying how sure we are.

Two very different situations get conflated by tools that just say "detected":

* **HDR is genuinely tagged.** PQ and HLG files carry ``smpte2084`` or
  ``arib-std-b67`` in their transfer characteristics, and that tag is reliable.
* **LOG almost never is.** A camera recording S-Log3 writes ``bt709`` into the
  file, because there is no standard tag for it. Nothing in the metadata
  distinguishes S-Log3 from ordinary Rec.709 footage - only the picture does,
  and only a human can judge that.

So this module returns a suggestion *with* a confidence, and the UI shows the
confidence rather than hiding it.  A guess presented as a fact is worse than no
guess at all: it produces a wrong conversion that looks deliberate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..core.camera_profiles import PROFILES, SourceProfile, get_profile
from ..core.colorimetry import COLOR_SPACE_ALIASES, COLOR_SPACES
from ..core.transfer import TRANSFER_ALIASES, TRANSFERS
from .probe import MediaInfo

#: Confidence levels, in increasing order of trust.
UNKNOWN = "unknown"
GUESS = "guess"
LIKELY = "likely"
TAGGED = "tagged"

_CONFIDENCE_LABEL = {
    TAGGED: "From file metadata",
    LIKELY: "Likely - inferred from the file",
    GUESS: "Guess - check this",
    UNKNOWN: "Not detected - defaulted",
}


@dataclass
class Detection:
    """A suggested interpretation plus an honest account of where it came from."""

    profile_id: str
    confidence: str = UNKNOWN
    reason: str = ""
    #: Other profiles worth trying, best first.
    alternatives: List[str] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []

    @property
    def profile(self) -> SourceProfile:
        return get_profile(self.profile_id)

    @property
    def confidence_label(self) -> str:
        return _CONFIDENCE_LABEL.get(self.confidence, self.confidence)

    @property
    def is_trustworthy(self) -> bool:
        """Whether the UI may apply this without drawing attention to it."""
        return self.confidence in (TAGGED, LIKELY)


def _normalise(value: str, aliases: dict, known: dict) -> str:
    key = (value or "").strip().lower().replace(" ", "")
    key = aliases.get(key, key)
    return key if key in known else ""


def normalise_transfer(value: str) -> str:
    """FFmpeg's transfer name to one of ours, or '' if unrecognised."""
    return _normalise(value, TRANSFER_ALIASES, TRANSFERS)


def normalise_primaries(value: str) -> str:
    return _normalise(value, COLOR_SPACE_ALIASES, COLOR_SPACES)


def _metadata_blob(info: MediaInfo) -> str:
    """Everything that might name a camera, lowercased for substring matching."""
    parts = [info.codec, info.codec_long, info.profile, info.container]
    parts.extend(info.tags.values())
    parts.append(info.path)
    return " ".join(str(p) for p in parts).lower()


def _camera_hint(info: MediaInfo) -> Optional[SourceProfile]:
    """Look for a camera name in the encoder/handler tags or the filename.

    This is genuinely just a hint - an S-Log3 clip that has been through a
    transcode carries whatever the transcoder wrote, and a file named
    'sony_wedding.mov' proves nothing about its curve.
    """
    blob = _metadata_blob(info)
    best: Optional[SourceProfile] = None
    best_len = 0
    for profile in PROFILES.values():
        for hint in profile.hints:
            if hint in blob and len(hint) > best_len:
                best, best_len = profile, len(hint)
    return best


def detect(info: MediaInfo) -> Detection:
    """Suggest a source interpretation for a probed file."""
    transfer = normalise_transfer(info.color_transfer)
    primaries = normalise_primaries(info.color_primaries)

    # 1. HDR is tagged, and the tag can be trusted.
    if transfer == "pq":
        profile = "hdr10" if primaries != "p3_d65" else "pq_p3"
        return Detection(profile, TAGGED,
                         f"Transfer tagged '{info.color_transfer}' (PQ) with "
                         f"{info.color_primaries or 'unspecified'} primaries.",
                         ["hdr10", "pq_p3"])
    if transfer == "hlg":
        profile = "hlg" if primaries != "bt709" else "hlg_709"
        return Detection(profile, TAGGED,
                         f"Transfer tagged '{info.color_transfer}' (HLG).",
                         ["hlg", "hlg_709"])

    # 2. A camera name in the metadata. Suggestive, never conclusive: the file
    #    is tagged Rec.709 either way, so we cannot tell LOG from a normal clip.
    hint = _camera_hint(info)
    if hint is not None and hint.group == "Camera LOG":
        return Detection(
            hint.id, GUESS,
            f"Metadata mentions this camera family, but LOG footage is always "
            f"tagged '{info.color_transfer or 'unspecified'}' - nothing in the "
            f"file confirms the curve. Check the picture before trusting this.",
            [hint.id, "rec709"],
        )

    # 3. Everything else: honour whatever the file claims.
    if transfer or primaries:
        transfer = transfer or "bt709"
        primaries = primaries or "bt709"
        from ..core.camera_profiles import find_profile
        match = find_profile(transfer, primaries)
        if match is not None:
            confidence = TAGGED if (info.color_transfer and info.color_primaries) else LIKELY
            return Detection(
                match.id, confidence,
                f"Tagged {info.color_transfer or 'no transfer'} / "
                f"{info.color_primaries or 'no primaries'}.",
                ["rec709", "srgb"],
            )

    return Detection(
        "rec709", UNKNOWN,
        "The file carries no usable colour tags. Defaulting to Rec.709 - if this "
        "is LOG or HDR footage you will need to set it by hand.",
        ["rec709", "slog3_sgamut3cine", "hdr10"],
    )


def suggested_peak_nits(info: MediaInfo, default: float = 1000.0) -> float:
    """Best available source peak, preferring what the file actually declares."""
    if info.mastering_peak_nits and info.mastering_peak_nits > 1.0:
        return float(info.mastering_peak_nits)
    if info.max_cll and info.max_cll > 1:
        return float(info.max_cll)
    transfer = normalise_transfer(info.color_transfer)
    if transfer == "hlg":
        return 1000.0     # HLG's nominal reference display.
    return float(default)
