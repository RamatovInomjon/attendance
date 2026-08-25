#!/usr/bin/env python3
"""Retention for recordings, debug frames and snapshots.

    python scripts/cleanup.py --dry-run       # show what would go
    python scripts/cleanup.py                 # apply configured retention
    python scripts/cleanup.py --free-gb 20    # also trim oldest until 20 GB free
"""
import argparse, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings

TARGETS = [
    ("recordings", settings.data_dir / "recordings", ("*.mp4", "*.json"), 30),
    ("debug frames", settings.debug_dir, ("*.jpg", "*.json"), 14),
    ("snapshots", settings.media_dir / "snapshots", ("*.jpg",), settings.snapshot_retention_days),
]


def files(root: Path, pats):
    for p in pats:
        yield from root.rglob(p)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--free-gb", type=float, default=0.0,
                   help="after age pruning, delete oldest until this much is free")
    args = a.parse_args()
    now = time.time()
    freed = 0

    for label, root, pats, days in TARGETS:
        if not root.exists():
            continue
        cutoff = now - days * 86400
        old = [f for f in files(root, pats) if f.is_file() and f.stat().st_mtime < cutoff]
        size = sum(f.stat().st_size for f in old)
        print(f"  {label:<15} keep {days:>3}d  ->  {len(old):>5} files, {size/1048576:8.1f} MB to remove")
        if not args.dry_run:
            for f in old:
                try:
                    f.unlink(); freed += f.stat().st_size if f.exists() else 0
                except OSError:
                    pass
            freed += size

    if args.free_gb:
        free = shutil.disk_usage(settings.data_dir).free / 1073741824
        if free < args.free_gb:
            allf = sorted((f for _l, r, p, _d in TARGETS if r.exists() for f in files(r, p) if f.is_file()),
                          key=lambda f: f.stat().st_mtime)
            print(f"\n  {free:.1f} GB free, want {args.free_gb} - trimming oldest")
            for f in allf:
                if shutil.disk_usage(settings.data_dir).free / 1073741824 >= args.free_gb:
                    break
                sz = f.stat().st_size
                if not args.dry_run:
                    try:
                        f.unlink(); freed += sz
                    except OSError:
                        pass

    print(f"\n  {'would free' if args.dry_run else 'freed'} {freed/1048576:.1f} MB")
    print(f"  disk now: {shutil.disk_usage(settings.data_dir).free/1073741824:.1f} GB free")


if __name__ == "__main__":
    main()
