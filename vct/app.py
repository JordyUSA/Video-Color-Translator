"""Application entry point."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .core.lut import DEFAULT_LUT_SIZE, MAX_LUT_SIZE


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vct",
        description="Interpret, tone map and export video colour.")
    parser.add_argument("file", nargs="?", help="video file to open on launch")
    parser.add_argument("--cpu", action="store_true",
                        help="force the CPU preview even if OpenGL is available")
    parser.add_argument("--lut-size", type=int, default=DEFAULT_LUT_SIZE,
                        help=f"3D LUT cube size, 2-{MAX_LUT_SIZE} "
                             f"(default {DEFAULT_LUT_SIZE}; 65 is more accurate "
                             f"on saturated wide-gamut footage)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not 2 <= args.lut_size <= MAX_LUT_SIZE:
        print(f"vct: --lut-size must be between 2 and {MAX_LUT_SIZE}",
              file=sys.stderr)
        return 2

    from PySide6.QtWidgets import QApplication

    from .ui.main_window import MainWindow
    from .ui.preview import configure_surface_format
    from .ui.theme import apply_theme

    # Must happen before the QApplication exists: Qt fixes the default surface
    # format at construction time.
    configure_surface_format()

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Video Color Translator")
    app.setApplicationDisplayName("Video Color Translator")
    apply_theme(app)

    window = MainWindow(initial_file=args.file, force_cpu_preview=args.cpu,
                        lut_size=args.lut_size)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
