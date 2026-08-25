#!/usr/bin/env python3
"""Compile the core algorithm to native extension modules.

    python scripts/build_native.py           # build .so next to the .py
    python scripts/build_native.py --clean   # remove build artefacts
    python scripts/build_native.py --verify  # build, then prove .so is imported

Why Cython rather than a C++ rewrite: measured, only ~21% of a frame is Python
at all - the other 79% is already inside OpenCV, numpy and ONNX Runtime. A hand
port would risk reintroducing bugs into code that currently measures d-prime
10.03 and 100% rank-1, to reclaim a fifth of a frame. Cython gets the same
distribution benefit (no shippable source) without touching the logic.

**This is obfuscation, not encryption.** A .so raises the cost of reading your
algorithm; it does not make it unreadable. Anyone with the machine can
disassemble it. Pair it with the licence check (app/core/licensing.py), and
understand that both together stop casual copying, not a determined analyst.

CPython imports an extension module in preference to a .py of the same name in
the same directory, so after a build the .so is what actually runs even with the
sources still present. `scripts/package_release.py` is what omits the sources.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "app" / "core"

# __init__.py stays interpreted: it is an empty package marker, and compiling a
# package's __init__ complicates the import machinery for no benefit.
SKIP = {"__init__.py"}


def sources() -> list[Path]:
    return sorted(p for p in CORE.glob("*.py") if p.name not in SKIP)


def clean() -> None:
    n = 0
    for pat in ("*.so", "*.c", "*.pyd"):
        for p in CORE.glob(pat):
            p.unlink()
            n += 1
    for d in (ROOT / "build",):
        if d.is_dir():
            shutil.rmtree(d)
    print(f"  removed {n} build artefact(s)")


def build(strip: bool = True) -> list[Path]:
    from setuptools import setup
    from Cython.Build import cythonize

    files = [str(p) for p in sources()]
    print(f"  compiling {len(files)} module(s) from app/core/")
    ext = cythonize(
        files,
        language_level="3",
        # binding=True keeps functions introspectable enough for dataclasses,
        # inspect.signature and FastAPI's dependency handling to keep working;
        # without it, decorated functions in the pipeline lose their signatures.
        compiler_directives={
            "binding": True,
            "embedsignature": False,   # do not bake signatures into docstrings
            "language_level": "3",
        },
        quiet=True,
    )
    argv = sys.argv
    sys.argv = [argv[0], "build_ext", "--inplace"]
    try:
        setup(name="ematsy-core", ext_modules=ext, script_args=["build_ext", "--inplace"])
    finally:
        sys.argv = argv

    built = sorted(CORE.glob("*.so"))
    if strip:
        for so in built:
            # Drop symbol names. Does not stop disassembly, but removes the
            # free map from addresses to function names.
            subprocess.run(["strip", "--strip-unneeded", str(so)],
                           check=False, capture_output=True)
    for so in built:
        print(f"    {so.name}  {so.stat().st_size/1024:.0f} KB")
    # The generated C is a full transliteration of the source - shipping it
    # would defeat the point entirely.
    for c in CORE.glob("*.c"):
        c.unlink()
    return built


def verify() -> int:
    """Import the pipeline and prove the modules came from .so, not .py."""
    sys.path.insert(0, str(ROOT))
    for name in ("app.core.geometry", "app.core.pipeline", "app.core.aligner",
                 "app.core.head_detector", "app.core.recognizer", "app.core.gallery"):
        mod = __import__(name, fromlist=["_"])
        origin = getattr(mod, "__file__", "?")
        kind = "NATIVE" if origin.endswith(".so") else "python"
        print(f"    {kind:7s} {name:26s} {Path(origin).name}")
        if kind != "NATIVE":
            print(f"      ^ still interpreted - the .so did not take precedence")
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-strip", action="store_true")
    args = ap.parse_args()

    if args.clean:
        clean()
        return 0

    build(strip=not args.no_strip)
    if args.verify:
        print("  verifying imports resolve to native modules:")
        return verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
