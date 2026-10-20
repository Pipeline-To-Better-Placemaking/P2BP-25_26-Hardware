# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller build recipe for mv3dt-installer (doc 00 §2, §4.1).
#
# Build (run from the repo root). The release job (doc 00 §4.1) runs:
#   pyinstaller installer/installer.spec --distpath dist \
#       --workpath /tmp/mv3dt-build --clean --noconfirm
# producing dist/mv3dt-installer at the repo root. The local build in
# installer/README.md section 8.3 passes --distpath installer/dist instead,
# producing installer/dist/mv3dt-installer. Either way the output is one
# self-contained executable.
#
# Bundled data is unpacked at runtime into sys._MEIPASS and resolved via
# mv3dt_installer.shellout.asset_path() (doc 00 §4.2). This spec ships two
# kinds of data:
#   1. Everything already under mv3dt_installer/assets/ (bundled bash
#      fragments, the tracked mosquitto/mv3dt.conf and cameras/cameras.yml
#      copies, and the systemd and polkit unit templates).
#   2. The DeepStream config templates that still live under laptop/
#      (tracker/infer/app/msgconv), pulled in directly from the laptop/
#      harness at build time so the single binary is self-contained
#      without duplicating those files by hand.
#
# NOTE: this file is executed by PyInstaller's own runtime, which injects
# `Analysis`, `PYZ`, `EXE`, and `SPECPATH` as globals before running it —
# it is not meant to run standalone under plain `python3 installer.spec`.

import os

block_cipher = None

# SPECPATH is injected by PyInstaller and points at the directory containing
# this .spec file (installer/). Anchor everything off it so the build works
# regardless of the invoking process's current working directory.
ROOT = os.path.dirname(os.path.abspath(SPECPATH))  # noqa: F821 (PyInstaller global)
INSTALLER_DIR = os.path.join(ROOT, "installer")
PACKAGE_DIR = os.path.join(INSTALLER_DIR, "mv3dt_installer")
ASSETS_DIR = os.path.join(PACKAGE_DIR, "assets")
LAPTOP_DEEPSTREAM_DIR = os.path.join(ROOT, "laptop", "deepstream")


def _collect_tree_datas(src_dir: str, dest_prefix: str) -> list[tuple[str, str]]:
    """Walk src_dir and return (source_file, dest_dir) pairs for `datas`,
    preserving the directory structure under dest_prefix. Skips .gitkeep
    placeholders — they exist only so the empty scaffolding dirs survive a
    fresh clone, not to be shipped inside the binary."""
    pairs: list[tuple[str, str]] = []
    if not os.path.isdir(src_dir):
        return pairs
    for dirpath, _dirnames, filenames in os.walk(src_dir):
        rel_dir = os.path.relpath(dirpath, src_dir)
        for filename in filenames:
            if filename == ".gitkeep":
                continue
            src_file = os.path.join(dirpath, filename)
            dest_dir = (
                dest_prefix
                if rel_dir == "."
                else os.path.join(dest_prefix, rel_dir)
            )
            pairs.append((src_file, dest_dir))
    return pairs


datas: list[tuple[str, str]] = []

# 1. installer/mv3dt_installer/assets/** -> assets/
datas += _collect_tree_datas(ASSETS_DIR, "assets")

# 2. DeepStream config templates from laptop/deepstream/ (tracker/infer/app/
#    msgconv) -> assets/deepstream/
for _fname in (
    "config_tracker_NvMOT.yml",
    "config_infer_primary.txt",
    "deepstream_app_config.txt",
    "msgconv_config.txt",
):
    _fpath = os.path.join(LAPTOP_DEEPSTREAM_DIR, _fname)
    if os.path.isfile(_fpath):
        datas.append((_fpath, "assets/deepstream"))

a = Analysis(  # noqa: F821 (PyInstaller global)
    [os.path.join(PACKAGE_DIR, "__main__.py")],
    pathex=[INSTALLER_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)  # noqa: F821

# --onefile: bundle everything (scripts + binaries + zipfiles + datas) into
# a single executable rather than emitting a COLLECT() directory build.
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="mv3dt-installer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # TUI/CLI-first, no X/Wayland dependency (doc 00 §2)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
