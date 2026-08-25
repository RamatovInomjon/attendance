#!/usr/bin/env python3
"""A/B camera settings against real faces, in one walking session.

Static-scene metrics cannot resolve the two settings that matter most here:

* **WDR** on these cameras is multi-exposure frame blending. On a static wall it
  is invisible; on a walking face it can ghost. Only motion reveals it.
* **Shutter** trades motion blur against gain-driven noise. The trade only shows
  up on a moving subject.

So this applies each configuration in turn while somebody keeps walking, and
measures what actually matters: the sharpness of aligned face crops and the
recognition scores they produce.

    python scripts/ab_image_test.py            # ~4 min, keep walking throughout
    python scripts/ab_image_test.py --window 90

Walk continuously past both cameras for the whole run. The script announces each
switch. Settings are restored to the starting values at the end.
"""
import argparse, json, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2, numpy as np
from sqlalchemy import select

from app.config import settings
from app.core.aligner import FaceAligner
from app.core.detector import build_detector
from app.core.quality import assess
from app.core.recognizer import FaceRecognizer
from app.core.stream import RtspSource
from app.core.tracker import FaceTracker
from app.db.models import Camera
from app.db.session import session_scope
from app.services.enrollment import load_gallery

USER, PWD = "admin", "@a123456"
NS = 'xmlns="http://www.hikvision.com/ver20/XMLSchema"'

CONFIGS = [
    # name,                       wdr,        shutter, dnr, sharp
    ("A baseline  WDR50 1/250",   ("open",50), "1/250", 10, 60),
    ("B WDR OFF   1/250",         ("close",50),"1/250", 15, 60),
    ("C WDR OFF   1/500",         ("close",50),"1/500", 15, 60),
    ("D WDR 30    1/500",         ("open",30), "1/500", 15, 60),
]


def put(ip, ep, xml):
    p = Path("/tmp/_ab.xml"); p.write_text(xml)
    r = subprocess.run(["curl","-s","-m","15","--digest","-u",f"{USER}:{PWD}","-X","PUT",
        "-H","Content-Type: application/xml","--data-binary",f"@{p}",
        f"http://{ip}/ISAPI/{ep}"], capture_output=True, text=True)
    return "OK" in r.stdout


def get(ip, ep):
    r = subprocess.run(["curl","-s","-m","8","--digest","-u",f"{USER}:{PWD}",
        f"http://{ip}/ISAPI/{ep}"], capture_output=True, text=True)
    return r.stdout


def apply(ip, wdr, shutter, dnr, sharp):
    mode, level = wdr
    ok = True
    ok &= put(ip, "Image/channels/1/WDR",
        f'<?xml version="1.0" encoding="UTF-8"?><WDR version="2.0" {NS}>'
        f'<mode>{mode}</mode><WDRLevel>{level}</WDRLevel></WDR>')
    ok &= put(ip, "Image/channels/1/shutter",
        f'<?xml version="1.0" encoding="UTF-8"?><Shutter version="2.0" {NS}>'
        f'<ShutterLevel>{shutter}</ShutterLevel></Shutter>')
    ok &= put(ip, "Image/channels/1/noiseReduce",
        f'<?xml version="1.0" encoding="UTF-8"?><NoiseReduce version="2.0" {NS}>'
        f'<mode>general</mode><GeneralMode><generalLevel>{dnr}</generalLevel>'
        f'</GeneralMode></NoiseReduce>')
    ok &= put(ip, "Image/channels/1/sharpness",
        f'<?xml version="1.0" encoding="UTF-8"?><Sharpness version="2.0" {NS}>'
        f'<SharpnessLevel>{sharp}</SharpnessLevel></Sharpness>')
    return ok


def snapshot_settings(ip):
    w = get(ip, "Image/channels/1/WDR")
    import re
    mode = re.search(r"<mode>([^<]+)", w)
    lvl = re.search(r"<WDRLevel>([^<]+)", w)
    sh = re.search(r"<ShutterLevel>([^<]+)", get(ip, "Image/channels/1/shutter"))
    dn = re.search(r"<generalLevel>([^<]+)", get(ip, "Image/channels/1/noiseReduce"))
    sp = re.search(r"<SharpnessLevel>([^<]+)", get(ip, "Image/channels/1/sharpness"))
    return ((mode.group(1) if mode else "open", int(lvl.group(1)) if lvl else 50),
            sh.group(1) if sh else "1/250",
            int(dn.group(1)) if dn else 10,
            int(sp.group(1)) if sp else 60)


class Collector(threading.Thread):
    """Runs the real pipeline on one camera and records per-face measurements."""
    def __init__(self, cam, gallery):
        super().__init__(daemon=True)
        self.cid, self.name, self.url = cam
        self.g = gallery
        self.bucket = None
        self.rows = []
        self.stop_flag = threading.Event()

    def run(self):
        det = build_detector("yolo", settings.model_path(settings.detector_model),
                             imgsz=settings.detect_width, conf=settings.detect_conf)
        ali = FaceAligner(settings.model_path(settings.aligner_model),
                          crop_size=settings.align_crop_size,
                          margin=settings.align_margin, mode=settings.align_mode)
        rec = FaceRecognizer(settings.model_path(settings.recognizer_model))
        trk = FaceTracker(frame_rate=12)
        src = RtspSource(self.url, name=self.name, transport="tcp", queue_size=2).start()
        t0 = time.time()
        while not src.connected and time.time()-t0 < 20: time.sleep(0.3)
        nth = 0
        while not self.stop_flag.is_set():
            f = src.read(timeout=2.0)
            if f is None: continue
            nth += 1
            if nth % 2: continue
            if self.bucket is None: continue
            h, w = f.image.shape[:2]
            sc = settings.detect_width / w
            dets = det.detect(cv2.resize(f.image,(settings.detect_width,int(h*sc))))
            if not dets:
                trk.update(np.zeros((0,4),np.float32), np.zeros((0,),np.float32)); continue
            boxes = np.stack([d.box for d in dets])/sc
            tracked = trk.update(boxes, np.array([d.score for d in dets],np.float32))
            if not tracked: continue
            rgb = cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB)
            faces = ali.align(rgb, [b for _t,b,_s in tracked])
            if not faces: continue
            embs = rec.embed(np.stack([x.aligned for x in faces]))
            for (tid,b,_s), af, emb in zip(tracked, faces, embs):
                q = assess(b, af.aligned, af.landmarks, af.score,
                    min_face_px=settings.min_face_px, min_laplacian_var=settings.min_laplacian_var,
                    min_aligner_score=settings.min_aligner_score,
                    max_yaw_deg=settings.max_yaw_deg, max_pitch_deg=settings.max_pitch_deg)
                sims = self.g.M @ (emb/(np.linalg.norm(emb)+1e-12))
                per = {}
                for o,v in zip(self.g.owner, sims): per[int(o)] = max(per.get(int(o),-2), float(v))
                best = max(per.values()) if per else 0.0
                self.rows.append(dict(cfg=self.bucket, cam=self.name, sharp=q.sharpness,
                                      face_px=q.face_px, gate=q.reason or "PASS", best=best))
        src.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=70, help="seconds per configuration")
    a = ap.parse_args()

    with session_scope() as s:
        cams = [(c.id, c.name, c.rtsp_url) for c in
                s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars()]
        ips = [c.ip for c in s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars()]

    original = {ip: snapshot_settings(ip) for ip in ips}
    print("original settings:")
    for ip, o in original.items(): print(f"   {ip}: WDR={o[0]} shutter={o[1]} DNR={o[2]} sharp={o[3]}")

    g = load_gallery()
    cols = [Collector(c, g) for c in cams]
    for c in cols: c.start()
    time.sleep(6)

    print(f"\n{'='*72}\nWALK CONTINUOUSLY past both cameras for the next "
          f"{len(CONFIGS)*(a.window+12)//60} minutes.\n{'='*72}\n")
    try:
        for name, wdr, sh, dn, sp in CONFIGS:
            for ip in ips: apply(ip, wdr, sh, dn, sp)
            print(f"  >>> {name}   (settling 12s, then {a.window}s of capture)")
            time.sleep(12)
            for c in cols: c.bucket = name
            time.sleep(a.window)
            for c in cols: c.bucket = None
            n = sum(1 for c in cols for r in c.rows if r["cfg"] == name)
            print(f"      collected {n} face observations")
    finally:
        for ip, o in original.items(): apply(ip, *o)
        for c in cols: c.stop_flag.set()
        print("\n  settings restored to the starting values")

    rows = [r for c in cols for r in c.rows]
    Path("data").mkdir(exist_ok=True)
    Path("data/ab_image_test.json").write_text(json.dumps(rows, indent=1))
    if not rows:
        print("\nNo faces captured - nobody walked past."); return

    print(f"\n{'='*94}\nRESULTS  ({len(rows)} face observations)\n{'='*94}")
    print(f"{'config':<26} {'n':>4} {'sharp med':>10} {'sharp p75':>10} "
          f"{'gate-pass':>10} {'best med':>9} {'best max':>9}")
    print("-"*94)
    for name, *_ in CONFIGS:
        sub = [r for r in rows if r["cfg"] == name]
        if not sub:
            print(f"{name:<26} {'0':>4}  no data"); continue
        sh_ = np.array([r["sharp"] for r in sub])
        bs = np.array([r["best"] for r in sub])
        gp = sum(1 for r in sub if r["gate"] == "PASS")/len(sub)*100
        print(f"{name:<26} {len(sub):>4} {np.median(sh_):>10.0f} {np.percentile(sh_,75):>10.0f} "
              f"{gp:>9.0f}% {np.median(bs):>9.3f} {bs.max():>9.3f}")
    print("\nHigher sharpness and higher best-score are better. Watch gate-pass too:")
    print("it is the fraction of observations usable for recognition at all.")


if __name__ == "__main__":
    main()
