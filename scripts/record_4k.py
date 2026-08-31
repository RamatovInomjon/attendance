#!/usr/bin/env python3
"""Record native-4K clips of people passing, for offline algorithm work.

    python scripts/record_4k.py --minutes 60
    python scripts/record_4k.py --camera Entrance --minutes 30
    python scripts/record_4k.py --minutes 60 --pre 5 --post 5

Why this exists rather than `record_clips=1 python scripts/run.py`: the in-app
`ClipRecorder` buffers DECODED frames for its pre-roll and re-encodes them, and
`record_width=1920` halves every clip. That halving is what invalidated the
face-size gate sweep - a head box in those recordings is half its live size, so
no gate, threshold or score measured on them transfers to production.

Two constraints shape the design, and both rule out the obvious approach:

* **RAM.** A 5 s pre-roll at 4K/20 fps is 100 frames x 24.9 MB = 2.5 GB per
  camera. This machine has ~2 GB available with 5 GB already swapped. Buffering
  decoded frames is not an option.
* **Fidelity.** Re-encoding 4K costs CPU and throws away exactly the detail the
  recordings are meant to preserve.

So nothing is decoded on the recording path at all. ffmpeg stream-copies each
camera into short segments on disk - a demux, no decode, no encode, no frame
buffer - and a clip is assembled by concatenating the segments that span the
event. The pre-roll comes from segments already written, so it costs no memory.

The trigger runs on the camera's 640x360 SUBSTREAM (channel 102), which is cheap
to decode, using the same head/person detector the pipeline uses. The main
stream is never decoded here.

This writes NOTHING to the database and makes no attendance decision. It only
reads the camera rows for their URLs, so it is safe to run while the production
service on the GPU server is capturing the same cameras.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("record4k")

SEGMENT_S = 2          # keyframe-aligned cut granularity; GOP is 12 @ 20 fps
RING_KEEP_S = 60       # how far back the ring is allowed to reach


def _stamp(p: Path) -> float:
    """Wall-clock start of a segment, from its %Y%m%d_%H%M%S name."""
    from app.config import settings
    return datetime.strptime(p.stem, "%Y%m%d_%H%M%S").replace(
        tzinfo=settings.tz).timestamp()


class Segmenter:
    """ffmpeg stream-copying one camera into a ring of short segments."""

    def __init__(self, name: str, url: str, ring: Path):
        self.name, self.url, self.ring = name, url, ring
        ring.mkdir(parents=True, exist_ok=True)
        for old in ring.glob("*.mp4"):
            old.unlink()
        self.proc: subprocess.Popen | None = None

    def start(self):
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp", "-timeout", "10000000",
            "-i", self.url,
            "-an", "-c", "copy",
            "-f", "segment", "-segment_time", str(SEGMENT_S),
            "-reset_timestamps", "1", "-strftime", "1",
            str(self.ring / "%Y%m%d_%H%M%S.mp4"),
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        log.info("[%s] segmenter started (stream copy, no re-encode)", self.name)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def segments(self) -> list[tuple[float, Path]]:
        out = []
        for p in self.ring.glob("*.mp4"):
            try:
                out.append((_stamp(p), p))
            except ValueError:
                continue
        return sorted(out)

    def prune(self, before: float):
        """Drop segments the pre-roll can no longer need.

        The NEWEST segment is never removed: ffmpeg is still writing it, and
        unlinking an open output leaves the muxer writing to a deleted inode.
        """
        segs = self.segments()
        for ts, p in segs[:-1]:
            if ts + SEGMENT_S < before:
                p.unlink(missing_ok=True)

    def stop(self):
        if self.alive():
            self.proc.send_signal(signal.SIGINT)   # let it finalise the mp4
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class Trigger:
    """Person/head detection on the 640x360 substream. Cheap, and never
    touches the 4K stream."""

    def __init__(self, name: str, sub_url: str, every_nth: int, conf: float):
        self.name, self.url, self.every_nth = name, sub_url, every_nth
        self.conf = conf
        self.active = False
        self.last_seen = 0.0
        self.first_seen = 0.0
        self.frames = 0
        self.detections = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name=f"trig-{self.name}",
                                        daemon=True)
        self._thread.start()

    def _run(self):
        import cv2
        import numpy as np
        from app.config import settings
        from app.core.head_detector import HeadDetector

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        det = HeadDetector(settings.model_path(settings.head_model),
                           size=settings.head_input, conf=self.conf)
        log.info("[%s] trigger on %s", self.name, det.provider)

        cap = None
        n = 0
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    self.error = "substream would not open"
                    time.sleep(2.0)
                    continue
                self.error = None
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                continue
            self.frames += 1
            n += 1
            if n % self.every_nth:
                continue
            hits = det.detect(frame, want=None)
            now = time.time()
            if hits:
                self.detections += 1
                if not self.active:
                    self.first_seen = now
                    self.active = True
                self.last_seen = now
        if cap is not None:
            cap.release()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)


def concat(segs: list[Path], out: Path) -> bool:
    """Join segments without re-encoding."""
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in segs))
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c", "copy", str(out)],
        capture_output=True)
    listing.unlink(missing_ok=True)
    if r.returncode != 0:
        log.error("concat failed: %s", r.stderr.decode()[:200])
        out.unlink(missing_ok=True)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--camera", action="append", default=None,
                    help="camera name; repeatable. Default: every enabled camera")
    ap.add_argument("--pre", type=float, default=5.0, help="seconds before")
    ap.add_argument("--post", type=float, default=5.0, help="seconds after")
    ap.add_argument("--conf", type=float, default=None,
                    help="detector confidence (default: settings.head_conf)")
    ap.add_argument("--every-nth", type=int, default=3,
                    help="detect on every Nth substream frame (20 fps / 3 = ~7 Hz)")
    ap.add_argument("--out", default=None, help="default data/recordings_4k")
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    ap.add_argument("--max-clip", type=float, default=120.0,
                    help="force-close a clip after this long; somebody standing "
                         "in the corridor would otherwise record forever")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    from app.config import settings
    from app.core.onnx_env import preload_cuda_libs
    preload_cuda_libs()
    import sqlite3

    db = ROOT / "data" / "ematsy.db"
    rows = sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute(
        "select name, rtsp_url from camera where enabled=1 order by id").fetchall()
    if args.camera:
        want = {c.lower() for c in args.camera}
        rows = [r for r in rows if r[0].lower() in want]
    if not rows:
        raise SystemExit("  no matching enabled cameras")

    out_root = Path(args.out) if args.out else settings.data_dir / "recordings_4k"
    ring_root = settings.data_dir / "_ring4k"
    conf = settings.head_conf if args.conf is None else args.conf

    cams = []
    for name, url in rows:
        sub = url.replace("/Streaming/Channels/101", "/Streaming/Channels/102")
        seg = Segmenter(name, url, ring_root / name)
        trg = Trigger(name, sub, args.every_nth, conf)
        (out_root / name).mkdir(parents=True, exist_ok=True)
        cams.append({"name": name, "seg": seg, "trg": trg,
                     "out": out_root / name, "clips": 0, "event": None})

    print(f"\n  recording NATIVE 4K, stream copy - no re-encode, no frame buffer")
    print(f"  cameras : {', '.join(c['name'] for c in cams)}")
    print(f"  padding : {args.pre:.0f}s before / {args.post:.0f}s after a person")
    print(f"  output  : {out_root}")
    print(f"  for     : {args.minutes:.0f} min   (Ctrl-C to stop early)\n")

    for c in cams:
        c["seg"].start()
        c["trg"].start()

    deadline = time.time() + args.minutes * 60
    stop = threading.Event()

    def _sig(*_):
        print("\n  stopping...")
        stop.set()
    signal.signal(signal.SIGINT, _sig)

    written = 0
    try:
        while not stop.is_set() and time.time() < deadline:
            time.sleep(1.0)
            now = time.time()
            free = shutil.disk_usage(out_root).free / 1073741824
            if free < args.min_free_gb:
                log.warning("only %.1f GB free (need %.1f) - stopping", free,
                            args.min_free_gb)
                break

            for c in cams:
                seg, trg = c["seg"], c["trg"]
                if not seg.alive():
                    log.warning("[%s] segmenter died, restarting", c["name"])
                    seg.start()

                if trg.active and c["event"] is None:
                    c["event"] = trg.first_seen
                    log.info("[%s] person in view", c["name"])

                # The event ends once nobody has been seen for the post-roll,
                # or once it has run long enough to be somebody loitering rather
                # than passing.
                gone = trg.active and now - trg.last_seen > args.post
                over = c["event"] is not None and now - c["event"] > args.max_clip
                if c["event"] is not None and (gone or over):
                    end = trg.last_seen if gone else now
                    t0, t1 = c["event"] - args.pre, end + args.post
                    c["event"] = None
                    if gone:
                        trg.active = False
                    else:
                        trg.first_seen = now      # keep recording as a new clip
                        log.info("[%s] clip capped at %.0fs, continuing",
                                 c["name"], args.max_clip)

                    # Wait for the segment covering t1 to be closed by ffmpeg.
                    time.sleep(SEGMENT_S + 1.0)
                    picked = [p for ts, p in seg.segments()
                              if ts + SEGMENT_S >= t0 and ts <= t1]
                    if not picked:
                        log.warning("[%s] no segments for the event", c["name"])
                        continue
                    stamp = datetime.fromtimestamp(t0, tz=settings.tz)
                    dest = c["out"] / f"{stamp:%Y%m%d_%H%M%S}_{c['name']}.mp4"
                    if concat(picked, dest):
                        mb = dest.stat().st_size / 1e6
                        c["clips"] += 1
                        written += 1
                        dest.with_suffix(".json").write_text(json.dumps({
                            "camera": c["name"], "start": t0, "end": t1,
                            "duration_s": round(t1 - t0, 1),
                            "segments": len(picked),
                            "pre_roll_s": args.pre, "post_roll_s": args.post,
                            "resolution": "native (stream copy)",
                            "trigger": "head/person detector on ch102 substream",
                            "detector_conf": conf,
                        }, indent=2))
                        log.info("[%s] clip %s  %.0fs  %.0f MB", c["name"],
                                 dest.name, t1 - t0, mb)

                seg.prune(before=now - max(RING_KEEP_S, args.pre + 4 * SEGMENT_S))
    finally:
        for c in cams:
            c["trg"].stop()
            c["seg"].stop()
        shutil.rmtree(ring_root, ignore_errors=True)

    print(f"\n  {'=' * 60}")
    for c in cams:
        t = c["trg"]
        print(f"  {c['name']:10s} {c['clips']:3d} clips   "
              f"substream {t.frames} frames, {t.detections} with people"
              + (f"   [{t.error}]" if t.error else ""))
    print(f"  {written} clips in {out_root}")
    print(f"  {'=' * 60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
