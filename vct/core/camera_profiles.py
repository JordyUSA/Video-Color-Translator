"""Named source interpretations: a transfer curve plus the gamut that goes with it.

A camera LOG format is never just a curve - S-Log3 footage is S-Gamut3.cine
footage too, and interpreting one without the other leaves the image the wrong
kind of wrong.  This module is the list the "Interpret As" dropdown is built
from, so each entry pairs both and carries the note the UI shows underneath.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SourceProfile:
    id: str
    label: str
    transfer: str
    primaries: str
    group: str            # "Standard" | "HDR" | "Camera LOG"
    note: str = ""
    #: Substrings that, seen in codec/encoder/handler metadata, hint at this profile.
    hints: tuple = ()


PROFILES: Dict[str, SourceProfile] = {p.id: p for p in [
    # ---------------- Standard ----------------
    SourceProfile("rec709", "Rec.709 (HD video)", "bt709", "bt709", "Standard",
                  "The default interpretation for ordinary HD/UHD SDR footage."),
    SourceProfile("srgb", "sRGB", "srgb", "bt709", "Standard",
                  "Screen recordings, renders and stills-derived footage."),
    SourceProfile("gamma24_709", "Rec.709 / Gamma 2.4 (BT.1886)", "bt1886", "bt709", "Standard",
                  "Footage already graded on a reference monitor."),
    SourceProfile("rec601_625", "Rec.601 625-line (PAL/SD)", "bt709", "bt601_625", "Standard"),
    SourceProfile("rec601_525", "Rec.601 525-line (NTSC/SD)", "bt709", "bt601_525", "Standard"),
    SourceProfile("p3_d65", "Display P3", "srgb", "p3_d65", "Standard",
                  "Apple-ecosystem SDR delivery."),
    SourceProfile("dcip3", "DCI-P3 (Gamma 2.6)", "gamma26", "p3_dci", "Standard"),
    SourceProfile("linear_709", "Linear (Rec.709 primaries)", "linear", "bt709", "Standard",
                  "Scene-linear EXR-style footage."),

    # ---------------- HDR ----------------
    SourceProfile("hdr10", "HDR10 / PQ (Rec.2020)", "pq", "bt2020", "HDR",
                  "SMPTE ST 2084 absolute HDR. The usual HDR10 / Dolby Vision base layer.",
                  hints=("smpte2084", "pq")),
    SourceProfile("pq_p3", "PQ (P3-D65 primaries)", "pq", "p3_d65", "HDR",
                  "PQ mastered on a P3 display but carried in a Rec.2020 container."),
    SourceProfile("hlg", "HLG (Rec.2020)", "hlg", "bt2020", "HDR",
                  "Broadcast HDR. Relative, display-adaptive.",
                  hints=("arib-std-b67", "hlg")),
    SourceProfile("hlg_709", "HLG (Rec.709 primaries)", "hlg", "bt709", "HDR"),

    # ---------------- Camera LOG ----------------
    SourceProfile("slog3_sgamut3cine", "Sony S-Log3 / S-Gamut3.cine", "slog3",
                  "s_gamut3_cine", "Camera LOG",
                  "Sony FX3/FX6/FX9/A7S III and Venice. The most common Sony setting.",
                  hints=("sony", "xavc", "s-log3", "slog3")),
    SourceProfile("slog3_sgamut3", "Sony S-Log3 / S-Gamut3", "slog3", "s_gamut3",
                  "Camera LOG", "The wider Sony gamut; less common than S-Gamut3.cine.",
                  hints=("sony", "s-gamut3")),
    SourceProfile("vlog_vgamut", "Panasonic V-Log / V-Gamut", "vlog", "v_gamut",
                  "Camera LOG", "Lumix S5/S1H/GH5-6 and Varicam.",
                  hints=("panasonic", "lumix", "v-log", "vlog")),
    SourceProfile("clog3_cinemagamut", "Canon C-Log3 / Cinema Gamut", "clog3",
                  "cinema_gamut", "Camera LOG", "Canon C70/C300/R5C in Cinema Gamut.",
                  hints=("canon", "clog3", "c-log3")),
    SourceProfile("clog3_bt2020", "Canon C-Log3 / Rec.2020", "clog3", "bt2020",
                  "Camera LOG", "Canon mirrorless bodies usually record C-Log3 in BT.2020.",
                  hints=("canon", "eos")),
    SourceProfile("clog2_cinemagamut", "Canon C-Log2 / Cinema Gamut", "clog2",
                  "cinema_gamut", "Camera LOG",
                  "Canon's widest curve; expects a grade rather than a simple LUT."),
    SourceProfile("nlog_bt2020", "Nikon N-Log / Rec.2020", "nlog", "bt2020",
                  "Camera LOG", "Nikon Z6/Z7/Z8/Z9 internal and ProRes RAW-derived N-Log.",
                  hints=("nikon", "n-log", "nlog")),
    SourceProfile("logc3_awg3", "ARRI LogC3 / ARRI Wide Gamut 3", "logc3", "awg3",
                  "Camera LOG", "ALEXA Classic, Mini, LF, Amira. EI 800 curve.",
                  hints=("arri", "alexa", "logc", "amira")),
    SourceProfile("applelog_bt2020", "Apple Log / Rec.2020", "applelog", "bt2020",
                  "Camera LOG", "iPhone 15 Pro and later, ProRes with Apple Log.",
                  hints=("apple", "iphone", "applelog")),
    SourceProfile("bmdfilm5_bmdwg", "Blackmagic Film Gen 5 / BMD Wide Gamut", "bmdfilm5",
                  "bmdwg", "Camera LOG", "Pocket 4K/6K and URSA in Gen 5 colour science.",
                  hints=("blackmagic", "bmd", "ursa", "pocket")),
]}

GROUP_ORDER: List[str] = ["Standard", "HDR", "Camera LOG"]


def get_profile(profile_id: str) -> SourceProfile:
    key = str(profile_id).strip().lower()
    if key not in PROFILES:
        raise KeyError(f"unknown source profile {profile_id!r}")
    return PROFILES[key]


def profiles_by_group() -> Dict[str, List[SourceProfile]]:
    out: Dict[str, List[SourceProfile]] = {g: [] for g in GROUP_ORDER}
    for p in PROFILES.values():
        out.setdefault(p.group, []).append(p)
    return out


def find_profile(transfer: str, primaries: str) -> Optional[SourceProfile]:
    """First profile matching a transfer/primaries pair, or None."""
    for p in PROFILES.values():
        if p.transfer == transfer and p.primaries == primaries:
            return p
    return None


# --------------------------------------------------------------------------
# Output (delivery) targets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputProfile:
    id: str
    label: str
    transfer: str
    primaries: str
    #: FFmpeg tagging for the encoded file.
    ff_primaries: str
    ff_transfer: str
    ff_matrix: str
    note: str = ""


OUTPUT_PROFILES: Dict[str, OutputProfile] = {o.id: o for o in [
    OutputProfile("rec709", "Rec.709 (SDR)", "bt709", "bt709",
                  "bt709", "bt709", "bt709",
                  "Standard SDR delivery for broadcast, YouTube and most players."),
    OutputProfile("srgb", "sRGB", "srgb", "bt709",
                  "bt709", "iec61966-2-1", "bt709",
                  "For web and computer displays. Slightly lifted shadows vs Rec.709."),
    OutputProfile("p3_d65", "Display P3", "srgb", "p3_d65",
                  "smpte432", "iec61966-2-1", "bt709",
                  "Wide-gamut SDR for Apple devices."),
    OutputProfile("rec709_g24", "Rec.709 / Gamma 2.4", "bt1886", "bt709",
                  "bt709", "bt709", "bt709",
                  "For a calibrated grading monitor in a dark room."),
]}


def get_output_profile(profile_id: str) -> OutputProfile:
    key = str(profile_id).strip().lower()
    if key not in OUTPUT_PROFILES:
        raise KeyError(f"unknown output profile {profile_id!r}")
    return OUTPUT_PROFILES[key]
