#!/usr/bin/env python3
"""Build the local gallery from face_id_users/.  Usage: python scripts/enroll.py"""
import logging, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import init_db
from app.services.enrollment import Enroller, load_gallery

logging.basicConfig(level=logging.INFO, format="%(message)s")

def main():
    init_db()
    t0 = time.perf_counter()
    rep = Enroller().run()
    el = time.perf_counter() - t0

    print("\n" + "=" * 66)
    print(f"  {rep.summary()}")
    print(f"  {el:.1f}s  ({el / max(rep.embedded, 1) * 1000:.0f} ms/image)")
    print("=" * 66)
    for label, items in (("no face detected", rep.no_face), ("alignment failed", rep.no_align)):
        if items:
            print(f"\n  {label}:")
            for i in items: print(f"    {i}")
    if rep.outliers:
        print("\n  outliers (disagree with their own folder — review these):")
        for name, c in sorted(rep.outliers, key=lambda x: x[1]):
            print(f"    {c:.3f}  {name}")

    g = load_gallery()
    print(f"\n  gallery loaded: {len(g)} embeddings / {g.n_people} people\n")

if __name__ == "__main__":
    main()
