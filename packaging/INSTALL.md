# Installing on Linux

## Dependencies

| Component | Why |
|---|---|
| Python 3.9+ | the application |
| FFmpeg 6.x / 7.x | decode, probe and encode. Needs the `lut3d`, `setparams` and `scale` filters — every distro build has them |
| PySide6 ≥ 6.5 | Qt 6 GUI (installed by pip) |
| NumPy ≥ 1.22 | colour maths (installed by pip) |
| OpenGL 3.3 | GPU preview. Optional — there is a CPU fallback |

### Ubuntu 22.04+ / Debian 12+

```bash
sudo apt install python3 python3-pip python3-venv ffmpeg \
                 libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0 \
                 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-xkb1
```

Qt 6.5+ needs `libxcb-cursor0`; without it the app exits with
"could not load the Qt platform plugin xcb".

### Arch / Manjaro

```bash
sudo pacman -S python python-pip ffmpeg qt6-base libxkbcommon-x11
```

### Fedora 38+

```bash
sudo dnf install python3 python3-pip mesa-libGL libxkbcommon-x11 xcb-util-cursor
sudo dnf install ffmpeg          # requires RPM Fusion
```

## Install

```bash
git clone https://github.com/JordyUSA/Video-Color-Translator.git
cd Video-Color-Translator
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
vct
```

## Desktop launcher

```bash
mkdir -p ~/.local/share/applications
sed "s|^Exec=vct|Exec=$(pwd)/.venv/bin/vct|" packaging/vct.desktop \
    > ~/.local/share/applications/vct.desktop
update-desktop-database ~/.local/share/applications
```

`Exec` is rewritten to the venv's binary so the launcher works without the
venv being activated.

## Verifying the install

```bash
vct --help
ffmpeg -filters | grep -E 'lut3d|setparams'      # both must appear
pip install -e '.[dev]' && pytest -q
```

## Troubleshooting

**"FFmpeg could not be located"** — install it, or point at a build:
`export VCT_FFMPEG=/opt/ffmpeg/bin/ffmpeg` and `VCT_FFPROBE` alongside it.

**"this ffmpeg build has no 'lut3d' filter"** — an unusually minimal build.
Install your distro's full `ffmpeg` package.

**Blank or black preview** — the OpenGL path could not initialise. The app
normally falls back on its own; `vct --cpu` forces it.

**Wayland issues** — `QT_QPA_PLATFORM=xcb vct` runs under XWayland.

**Slow preview without a GPU** — the CPU path draws a fast draft while a
slider moves and refines when it settles. A smaller window is materially
faster, since it renders at display resolution.
