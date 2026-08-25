#!/usr/bin/env python3
"""Pair every native face crop with the aligned 112x112 it produced.

Two images answer two different questions:

* **native** — the crop straight off the sensor, before alignment. If this is
  blurred, dark or noisy, the problem is the camera: exposure, focus, lighting,
  motion.
* **aligned** — the 112x112 the recognizer actually consumed. If the native crop
  looks fine but this one is skewed, cut off or off-centre, the problem is the
  detector box or the aligner, not the image.

Seeing them side by side is the only way to tell those apart. A low score with a
clean native crop and a bad aligned crop is a pipeline bug; a low score with
both looking poor is a camera problem.

    python scripts/view_faces.py                       # all sets found
    python scripts/view_faces.py data/ab_defaults      # one set
    xdg-open data/debug/faces.html
"""
import html, shutil, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

OUT = settings.debug_dir
SETS = ["data/walkthrough", "data/ab_defaults", "data/ab_mid", "data/ab_tuned"]


def collect(src: Path):
    """-> {stem: {'native': path|None, 'aligned': path|None, 'score': float}}"""
    items = defaultdict(dict)
    for p in sorted(src.glob("*.jpg")):
        n = p.name
        if n.endswith("_native.jpg"):
            stem, kind = n[:-len("_native.jpg")], "native"
        elif n.endswith("_face.jpg"):
            # live DebugCapture calls the native crop "_face"
            stem, kind = n[:-len("_face.jpg")], "native"
        elif n.endswith("_frame.jpg"):
            continue                              # whole frame, shown elsewhere
        elif n.endswith("_aligned.jpg"):
            stem, kind = n[:-len("_aligned.jpg")], "aligned"
        else:
            stem, kind = p.stem, "aligned"       # older runs saved aligned only
        items[stem][kind] = p
        try:
            items[stem]["score"] = float(stem.rsplit("_", 1)[1])
        except Exception:
            items[stem].setdefault("score", 0.0)
    return items


def main():
    srcs = [Path(a) for a in sys.argv[1:]] or (
        [Path(s) for s in SETS] +
        sorted(d for d in (settings.debug_dir).iterdir()
               if d.is_dir() and not d.name.startswith("faces_")))
    srcs = [s for s in srcs if s.is_dir() and any(s.glob("*.jpg"))]
    if not srcs:
        print("no crop sets found - run scripts/walkthrough.py first"); return

    OUT.mkdir(parents=True, exist_ok=True)
    sections = []
    for src in srcs:
        dest = OUT / f"faces_{src.name}"
        dest.mkdir(parents=True, exist_ok=True)
        items = collect(src)
        rows = []
        for stem, d in sorted(items.items(), key=lambda kv: -kv[1].get("score", 0)):
            cells = []
            for kind in ("native", "aligned"):
                p = d.get(kind)
                if p is None:
                    cells.append(f'<div class="miss">no {kind}</div>')
                    continue
                shutil.copy2(p, dest / p.name)
                cells.append(
                    f'<a href="faces_{src.name}/{p.name}" target="_blank">'
                    f'<img class="{kind}" src="faces_{src.name}/{p.name}" loading="lazy" alt=""></a>')
            sc = d.get("score", 0)
            cls = "good" if sc >= 0.30 else ("mid" if sc >= 0.20 else "low")
            rows.append(
                f'<figure class="pair {cls}">{cells[0]}{cells[1]}'
                f'<figcaption>{sc:.2f}<span>{html.escape(stem.rsplit("_",1)[0])}</span>'
                f'</figcaption></figure>')
        n_nat = sum(1 for d in items.values() if "native" in d)
        sections.append(
            f'<section><h2>{html.escape(src.name)}'
            f'<span class="n">{len(items)} observations · {n_nat} with native crops</span></h2>'
            f'<div class="grid">{"".join(rows)}</div></section>')
        print(f"  {src.name}: {len(items)} pairs ({n_nat} native)")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face crops</title><style>
:root{{--bg:#eef0f0;--sf:#f8f9f9;--ru:#c8d0d0;--ink:#0f1619;--ink3:#6b7a7d;
--good:#1f7a3d;--mid:#8a6a2e;--low:#a81e27}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0e1316;--sf:#151b1f;--ru:#2c373c;
--ink:#e2e8e8;--ink3:#7b8a8d;--good:#5fc07f;--mid:#cdae63;--low:#e9707a}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}}
.w{{max-width:1500px;margin:0 auto;padding:24px}}
h1{{font-size:1.5rem;letter-spacing:-.02em;margin:0 0 6px}}
.sub{{color:var(--ink3);font-size:.87rem;margin:0 0 24px;max-width:70ch}}
.sub b{{color:var(--ink)}}
section{{background:var(--sf);border:1px solid var(--ru);margin-bottom:18px}}
h2{{font-size:1rem;margin:0;padding:11px 16px;border-bottom:1px solid var(--ru);
display:flex;justify-content:space-between;align-items:baseline}}
.n{{font:600 .68rem/1 ui-monospace,monospace;letter-spacing:.1em;
text-transform:uppercase;color:var(--ink3)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;padding:14px}}
.pair{{margin:0;display:grid;grid-template-columns:1fr 1fr;gap:4px;
border:1px solid var(--ru);padding:4px;border-radius:3px}}
.pair img{{width:100%;aspect-ratio:1;object-fit:cover;display:block;background:#000}}
.pair img.aligned{{outline:2px solid var(--ink3);outline-offset:-2px}}
.pair.good{{border-color:var(--good)}}.pair.mid{{border-color:var(--mid)}}
.pair.low{{border-color:var(--low)}}
figcaption{{grid-column:1/-1;font:600 .72rem/1.4 ui-monospace,monospace;
display:flex;justify-content:space-between;padding:3px 2px 1px;color:var(--ink)}}
figcaption span{{color:var(--ink3);font-weight:400;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap;max-width:60%}}
.miss{{display:flex;align-items:center;justify-content:center;aspect-ratio:1;
background:var(--bg);color:var(--ink3);font:600 .66rem ui-monospace,monospace}}
</style></head><body><div class="w">
<h1>Face crops</h1>
<p class="sub"><b>Left = native crop</b> straight off the sensor, before alignment.
<b>Right = the aligned 112&times;112</b> the recognizer actually consumed (outlined).<br>
Blurry or dark on the left &rarr; camera problem. Clean left but skewed or
off-centre right &rarr; detector or aligner problem.<br>
Border colour is the match score: green &ge;0.30, amber &ge;0.20, red below.</p>
{"".join(sections)}
</div></body></html>"""
    (OUT / "faces.html").write_text(page)
    print(f"\n  open: {OUT / 'faces.html'}")


if __name__ == "__main__":
    main()
