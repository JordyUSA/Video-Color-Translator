"""3D LUT generation, .cube I/O, and a CPU application path.

The LUT is the hand-off between the colour engine and everything that renders:
the OpenGL preview uploads it as a 3D texture, and the exporter writes it to a
.cube for FFmpeg's ``lut3d`` filter.  Because both consume the same table, the
preview is not an approximation of the export - it is the same transform.

A 33-cube of float32 is 36 k samples, so rebuilding one while a slider is being
dragged costs a couple of milliseconds regardless of how large the video is.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .pipeline import ColorPipeline, PipelineParams

DEFAULT_LUT_SIZE = 33
MAX_LUT_SIZE = 129


def identity_lut(size: int = DEFAULT_LUT_SIZE) -> np.ndarray:
    """Identity table shaped ``(size, size, size, 3)``, indexed ``[b, g, r]``.

    That axis order is deliberate: it makes the C-order flattening put red
    fastest, which is both the .cube line order and the memory layout
    ``glTexImage3D`` wants for a width=R, height=G, depth=B texture.  One layout
    serves the file writer, the GPU upload and the CPU sampler.
    """
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)
    b, g, r = np.meshgrid(ramp, ramp, ramp, indexing="ij")
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def build_lut(params: PipelineParams, size: int = DEFAULT_LUT_SIZE) -> np.ndarray:
    """Bake a pipeline into a 3D LUT."""
    if not 2 <= size <= MAX_LUT_SIZE:
        raise ValueError(f"LUT size must be between 2 and {MAX_LUT_SIZE}, got {size}")
    grid = identity_lut(size)
    out = ColorPipeline(params).transform(grid.reshape(-1, 3))
    return np.asarray(out, dtype=np.float32).reshape(size, size, size, 3)


# --------------------------------------------------------------------------
# .cube I/O  (Adobe Cube LUT Specification 1.0)
# --------------------------------------------------------------------------

def write_cube(path: str, lut: np.ndarray, title: str = "Video Color Translator",
               domain_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
               domain_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> str:
    """Write a 3D LUT as a .cube file, red varying fastest."""
    lut = np.asarray(lut, dtype=np.float32)
    if lut.ndim != 4 or lut.shape[3] != 3 or len(set(lut.shape[:3])) != 1:
        raise ValueError(f"expected a cubic (N, N, N, 3) LUT, got {lut.shape}")
    size = lut.shape[0]
    flat = lut.reshape(-1, 3)

    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN {:.6f} {:.6f} {:.6f}".format(*domain_min),
        "DOMAIN_MAX {:.6f} {:.6f} {:.6f}".format(*domain_max),
        "",
    ]
    lines.extend("{:.6f} {:.6f} {:.6f}".format(*row) for row in flat)
    text = "\n".join(lines) + "\n"

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="ascii") as fh:
        fh.write(text)
    return path


def read_cube(path: str) -> Tuple[np.ndarray, dict]:
    """Read a 3D .cube file. Returns the LUT and its header fields.

    Lets the tool load a camera manufacturer's own conversion LUT instead of
    using the built-in curves, which is sometimes the only way to match a
    look a client has already approved.
    """
    size: Optional[int] = None
    title = ""
    dmin = [0.0, 0.0, 0.0]
    dmax = [1.0, 1.0, 1.0]
    values = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("TITLE"):
                title = line.split(None, 1)[1].strip().strip('"') if " " in line else ""
            elif upper.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
            elif upper.startswith("LUT_1D_SIZE"):
                raise ValueError(f"{path} is a 1D LUT; this tool needs a 3D .cube")
            elif upper.startswith("DOMAIN_MIN"):
                dmin = [float(v) for v in line.split()[1:4]]
            elif upper.startswith("DOMAIN_MAX"):
                dmax = [float(v) for v in line.split()[1:4]]
            else:
                parts = line.split()
                if len(parts) == 3:
                    values.append([float(v) for v in parts])

    if size is None:
        raise ValueError(f"{path} has no LUT_3D_SIZE header")
    expected = size ** 3
    if len(values) != expected:
        raise ValueError(f"{path}: expected {expected} entries for size {size}, "
                         f"found {len(values)}")
    lut = np.asarray(values, dtype=np.float32).reshape(size, size, size, 3)
    return lut, {"title": title, "size": size, "domain_min": tuple(dmin),
                 "domain_max": tuple(dmax)}


# --------------------------------------------------------------------------
# CPU application (preview fallback when there is no usable GL context)
# --------------------------------------------------------------------------

def apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinearly sample a 3D LUT over an image of shape ``(..., 3)``.

    Matches what the GPU does with a linearly-filtered 3D texture.  FFmpeg's
    tetrahedral interpolation differs by well under a code value on a smooth
    table this size.
    """
    image = np.asarray(image, dtype=np.float32)
    lut = np.asarray(lut, dtype=np.float32)
    size = lut.shape[0]
    shape = image.shape

    pos = np.clip(image.reshape(-1, 3), 0.0, 1.0) * (size - 1)
    i0 = np.floor(pos).astype(np.int32)
    np.clip(i0, 0, size - 2, out=i0)
    frac = (pos - i0).astype(np.float32)

    r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
    fr = frac[:, 0:1]
    fg = frac[:, 1:2]
    fb = frac[:, 2:3]

    flat = lut.reshape(-1, 3)

    def corner(dr: int, dg: int, db: int) -> np.ndarray:
        idx = ((b0 + db) * size + (g0 + dg)) * size + (r0 + dr)
        return flat[idx]

    c00 = corner(0, 0, 0) * (1 - fr) + corner(1, 0, 0) * fr
    c01 = corner(0, 0, 1) * (1 - fr) + corner(1, 0, 1) * fr
    c10 = corner(0, 1, 0) * (1 - fr) + corner(1, 1, 0) * fr
    c11 = corner(0, 1, 1) * (1 - fr) + corner(1, 1, 1) * fr

    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    return (c0 * (1 - fb) + c1 * fb).reshape(shape)
