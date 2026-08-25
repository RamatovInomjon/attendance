#!/usr/bin/env python3
"""Export every recognition on record into data/debug/ with a contact sheet.

Live captures land there automatically; this backfills from the database so
past events are reviewable too, and builds `index.html` - one page showing every
recognition grouped by person, newest first, so a wrong name is obvious at a
glance rather than something you go hunting for.

    python scripts/export_debug.py
    xdg-open data/debug/index.html
"""
import html, shutil, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.config import settings
from app.db.models import Camera, Employee, RecognitionEvent
from app.db.session import session_scope

OUT = settings.debug_dir
OUT.mkdir(parents=True, exist_ok=True)


def slug(n):
    keep = "".join(c if (c.isalnum() or c in " _-") else "_" for c in n).strip()
    return keep.replace(" ", "_") or "unknown"


def main():
    with session_scope() as s:
        rows = s.execute(
            select(RecognitionEvent, Employee.full_name, Employee.department, Camera.name)
            .join(Employee, Employee.id == RecognitionEvent.employee_id)
            .outerjoin(Camera, Camera.id == RecognitionEvent.camera_id)
            .order_by(RecognitionEvent.ts.desc())
        ).all()
        events = [{
            "name": n, "dept": dep or "", "camera": cam or "?",
            "ts": e.ts.astimezone(settings.tz), "score": e.score, "margin": e.margin,
            "role": e.role.value, "face_px": e.face_px, "transition": e.transition,
            "snapshot": e.snapshot, "track": e.track_id,
        } for e, n, dep, cam in rows]

    copied = 0
    for ev in events:
        if not ev["snapshot"]:
            continue
        src = settings.media_dir / ev["snapshot"]
        if not src.exists():
            continue
        d = OUT / slug(ev["name"]); d.mkdir(parents=True, exist_ok=True)
        dst = d / f"{ev['ts']:%Y%m%d_%H%M%S}_{ev['score']:.3f}_{ev['camera']}_db.jpg"
        if not dst.exists():
            shutil.copy2(src, dst); copied += 1

    # gather everything now in the folder, live captures included
    by_person = defaultdict(list)
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        for img in sorted(d.glob("*.jpg"), reverse=True):
            by_person[d.name].append(img)

    cards = []
    for person, imgs in sorted(by_person.items(), key=lambda kv: -len(kv[1])):
        aligned = [i for i in imgs if i.name.endswith("_aligned.jpg")]
        others = [i for i in imgs if not i.name.endswith("_aligned.jpg")]
        shown = (aligned + others)[:24]
        tiles = []
        for i in shown:
            parts = i.stem.split("_")
            sc = next((p for p in parts if p.replace(".", "").isdigit() and "." in p), "")
            rel = f"{person}/{i.name}"
            kind = ("aligned" if "_aligned" in i.name else
                    "frame" if "_frame" in i.name else
                    "face" if "_face" in i.name else "snapshot")
            tiles.append(
                f'<figure class="t {kind}"><a href="{rel}" target="_blank">'
                f'<img src="{rel}" loading="lazy" alt=""></a>'
                f'<figcaption>{sc} <span>{kind}</span></figcaption></figure>')
        cards.append(
            f'<section><h2>{html.escape(person.replace("_", " "))} '
            f'<span class="n">{len(imgs)} images</span></h2>'
            f'<div class="grid">{"".join(tiles)}</div></section>')

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Recognition debug</title><style>
:root{{--bg:#eef0f0;--sf:#f8f9f9;--ru:#c8d0d0;--ink:#0f1619;--ink3:#6b7a7d;--in:#0c6e73}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0e1316;--sf:#151b1f;--ru:#2c373c;--ink:#e2e8e8;--ink3:#7b8a8d;--in:#46b7b8}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}}
.w{{max-width:1400px;margin:0 auto;padding:24px}}
h1{{font-size:1.5rem;letter-spacing:-.02em;margin:0 0 4px}}
.sub{{color:var(--ink3);font-size:.86rem;margin:0 0 26px;
font-family:ui-monospace,monospace}}
section{{background:var(--sf);border:1px solid var(--ru);margin-bottom:18px}}
h2{{font-size:1rem;margin:0;padding:11px 16px;border-bottom:1px solid var(--ru);
display:flex;justify-content:space-between;align-items:baseline}}
.n{{font:600 .68rem/1 ui-monospace,monospace;letter-spacing:.1em;
text-transform:uppercase;color:var(--ink3)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));gap:10px;padding:14px}}
.t{{margin:0}}.t img{{width:100%;border:1px solid var(--ru);display:block;background:#000}}
.t.aligned img{{border-color:var(--in);border-width:2px}}
figcaption{{font:600 .66rem/1.4 ui-monospace,monospace;color:var(--ink3);
padding-top:4px;display:flex;justify-content:space-between}}
figcaption span{{opacity:.65}}
</style></head><body><div class="w">
<h1>Recognition debug</h1>
<p class="sub">{len(events)} events on record &middot; {sum(len(v) for v in by_person.values())} images &middot;
threshold {settings.recognition_threshold} &middot; align margin {settings.align_margin} ({settings.align_mode})<br>
teal border = the aligned 112&times;112 the recognizer actually saw</p>
{"".join(cards) or '<section><h2>Nothing captured yet</h2></section>'}
</div></body></html>"""
    (OUT / "index.html").write_text(page)

    print(f"  events on record : {len(events)}")
    print(f"  snapshots copied : {copied}")
    print(f"  people in folder : {len(by_person)}")
    for p, i in sorted(by_person.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"      {p:<32} {len(i)}")
    print(f"\n  open: {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
