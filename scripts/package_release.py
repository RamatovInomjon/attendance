#!/usr/bin/env python3
"""Build a deployable bundle: compiled core, encrypted models, no sources.

    python scripts/package_release.py --licence licence.key --out dist/

What goes in:
  * app/core/*.so            compiled, stripped - no .py for the core
  * app/{api,services,db,web} as .py (the web layer is not the IP)
  * models/*.onnx.enc        encrypted, useless without the licence
  * templates, static, scripts needed to run, docs

What is deliberately EXCLUDED, and why:
  * app/core/*.py            the algorithm in readable form
  * models/*.onnx            plaintext weights
  * models/_archive/         the fp32 master, which exists nowhere else
  * vendor_private_key.pem   mints licences for ANY machine
  * face_id_users/           biometric data - it belongs to the customer's
                             site, not in a shipped artefact
  * data/, .env, db          runtime state, secrets, and attendance records
  * tests/, bench/, .git

The bundle is per-customer: model encryption is keyed to their licence, so a
bundle built for one site will not run at another even with a valid licence of
its own.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INCLUDE_DIRS = ["app", "templates", "static", "docs", "deploy"]
INCLUDE_FILES = ["requirements.txt", "README.md", "run.sh", "env.example"]
INCLUDE_SCRIPTS = [
    "run.py", "seed_admin.py", "show_fingerprint.py", "enroll.py",
    "seed_cameras.py", "migrate.py", "set_direction.py", "session_report.py",
    "maintenance.py", "camera_config.py",
]

EXCLUDE_NAMES = {
    "__pycache__", ".pytest_cache", "_archive", ".git", "node_modules",
}
EXCLUDE_SUFFIXES = {".c", ".pyc", ".pyo"}


def _copy_app(dst: Path) -> tuple[int, int]:
    """Copy app/, dropping app/core/*.py so only the .so ships."""
    so_kept = py_dropped = 0
    for src in sorted((ROOT / "app").rglob("*")):
        if any(part in EXCLUDE_NAMES for part in src.parts):
            continue
        rel = src.relative_to(ROOT)
        if src.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
            continue
        if src.suffix in EXCLUDE_SUFFIXES:
            continue
        # The core ships compiled. __init__.py stays: it is an empty package
        # marker and the import system needs it.
        if src.parent.name == "core" and src.suffix == ".py" and src.name != "__init__.py":
            py_dropped += 1
            continue
        if src.suffix == ".so":
            so_kept += 1
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / rel)
    return so_kept, py_dropped


def _recognizer_is_shippable() -> list[str]:
    """The bundle must contain the recognizer it is configured to load.

    `_leaks()` below refuses plaintext .onnx, so a recognizer with no `.enc`
    ends up in the bundle in NEITHER form. The service then starts on the
    customer's machine, fails to construct FaceRecognizer, and dies - a failure
    discovered at their site rather than at build time. That is exactly what
    switching `recognizer_model` to a model that had never been encrypted
    would have produced.
    """
    from app.config import settings
    bad = []
    wanted = [settings.recognizer_model, settings.head_model,
              settings.aligner_model]
    if settings.reid_model:
        wanted.append(settings.reid_model)
    for name in wanted:
        if not (settings.models_dir / (str(name) + ".enc")).is_file():
            bad.append(f"{name} has no .enc (run scripts/encrypt_models.py)")
    return bad


def _debug_capture_is_off() -> list[str]:
    """Debug capture is for tuning, not for customer sites.

    Clip recording produced 3.2 GB of video of identified people in one day and
    save_all_frames wrote 15,724 face images. Shipping either enabled fills a
    customer's disk with biometric imagery they never asked for, so this is
    checked BEFORE anything is copied rather than after.
    """
    from app.config import settings as s
    hot = []
    if s.record_clips:
        hot.append("record_clips=True")
    if s.save_all_frames:
        hot.append("save_all_frames=True")
    if s.debug_max_per_person == 0:
        hot.append("debug_max_per_person=0 (unbounded)")
    return hot


def build(args) -> int:
    missing = _recognizer_is_shippable()
    if missing:
        print('  REFUSING: the configured models cannot be shipped:')
        for x in missing:
            print(f'    {x}')
        return 1
    hot = _debug_capture_is_off()
    if hot:
        print("  REFUSING TO SHIP - debug capture is enabled:", file=sys.stderr)
        for h in hot:
            print(f"    {h}", file=sys.stderr)
        print("  These are debug-only; unset them (or the matching env var) "
              "and try again.", file=sys.stderr)
        return 1

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    stage = out / "ematsy"
    stage.mkdir(parents=True)

    core_so = list((ROOT / "app" / "core").glob("*.so"))
    if not core_so:
        print("  no compiled core found. Run: python scripts/build_native.py",
              file=sys.stderr)
        return 1

    so_kept, py_dropped = _copy_app(stage)
    print(f"  app/: {so_kept} compiled module(s), {py_dropped} source file(s) withheld")

    for d in INCLUDE_DIRS[1:]:
        src = ROOT / d
        if src.is_dir():
            shutil.copytree(src, stage / d,
                            ignore=shutil.ignore_patterns(*EXCLUDE_NAMES))
    for f in INCLUDE_FILES:
        if (ROOT / f).is_file():
            shutil.copy2(ROOT / f, stage / f)

    (stage / "scripts").mkdir(exist_ok=True)
    for s in INCLUDE_SCRIPTS:
        if (ROOT / "scripts" / s).is_file():
            shutil.copy2(ROOT / "scripts" / s, stage / "scripts" / s)

    # models: encrypted only
    md = stage / "models"
    md.mkdir(exist_ok=True)
    enc = sorted((ROOT / "models").glob("*.onnx" + ".enc"))
    if not enc:
        print("  no encrypted models. Run:", file=sys.stderr)
        print("    python scripts/encrypt_models.py --licence <licence>", file=sys.stderr)
        return 1
    total = 0
    for e in enc:
        shutil.copy2(e, md / e.name)
        total += e.stat().st_size
    # the face detector used for enrolment is a public weight, ship as-is
    pt = ROOT / "models" / "yolov8n-face.pt"
    if pt.is_file():
        shutil.copy2(pt, md / pt.name)
        total += pt.stat().st_size
    print(f"  models/: {len(enc)} encrypted + face detector, {total/1e6:.0f} MB")

    if args.licence:
        shutil.copy2(args.licence, stage / "licence.key")
        print(f"  licence: {args.licence} included")
    else:
        print("  licence: NOT included - the customer installs their own")

    # a plaintext .onnx here would defeat the whole exercise
    leaked = [p for p in stage.rglob("*.onnx")]
    leaked += [p for p in (stage / "app" / "core").glob("*.py") if p.name != "__init__.py"]
    leaked += [p for p in stage.rglob("vendor_private_key.pem")]
    if leaked:
        print("  REFUSING TO SHIP - these would leak:", file=sys.stderr)
        for p in leaked:
            print(f"    {p.relative_to(stage)}", file=sys.stderr)
        return 1

    if args.tar:
        tar = out / "ematsy-release.tar.gz"
        with tarfile.open(tar, "w:gz") as t:
            t.add(stage, arcname="ematsy")
        print(f"  archive: {tar}  ({tar.stat().st_size/1e6:.0f} MB)")

    size = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print(f"  bundle : {stage}  ({size/1e6:.0f} MB)")
    print(f"  deploy instructions: docs/DEPLOYMENT.md")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dist")
    ap.add_argument("--licence", type=Path,
                    help="include this licence in the bundle")
    ap.add_argument("--tar", action="store_true", help="also produce a .tar.gz")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
