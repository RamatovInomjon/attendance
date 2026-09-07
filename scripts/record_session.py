#!/usr/bin/env python3
"""Record every camera continuously, at native 4K, for a fixed session.

    python scripts/record_session.py --hours 5
    python scripts/record_session.py --hours 5 --out /mnt/big/busy-day
    python scripts/record_session.py --minutes 2 --dry-run      # check first

Companion to `record_4k.py`, and the choice between them is not about disk:

* `record_4k.py` keeps only what its **head detector** triggered on. Whatever
  the detector missed was never written, so a recording made that way can
  never show you a miss. It is the wrong tool for building a set you intend to
  MEASURE the detector against - the answer is baked into the question.
* This script writes the stream unconditionally. Every pass is in there,
  including the ones the current pipeline fumbles, and the event clips can
  still be cut from it afterwards. Continuous -> clips is an offline edit;
  clips -> continuous is not recoverable.

So for a day whose whole point is "more traffic than usual, keep it for future
testing", this is the one to run.

Nothing is decoded. ffmpeg stream-copies each camera into short segments - a
demux, no decode, no encode, no frame buffer - so the cost is one socket and
one disk write per camera, and the bytes on disk are bit-identical to what the
recognizer would have received live. Re-encoding would throw away exactly the
compression artefacts and sensor detail that a face-recognition test set exists
to preserve.

The layout matches what the bench tools already glob (`<dir>/<Camera>/*.mp4`),
so a finished session feeds them with no conversion step:

    python bench/eval_4k.py --dir data/sessions/<stamp>
    python scripts/extract_persons.py --dir data/sessions/<stamp>

This writes NOTHING to the database and makes no attendance decision. It only
reads the camera rows for their URLs, so it is safe to run alongside the
production service capturing the same cameras.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("session")

GIB = 1073741824
RESTART_BACKOFF_S = 3.0


def _redact(text: str) -> str:
    """Strip RTSP URLs out of ffmpeg's chatter - they carry the password."""
    import re
    return re.sub(r"rtsp://\S*", "rtsp://<redacted>", text)


class Recorder:
    """One ffmpeg stream-copying a camera into timestamped segments.

    Restarts itself. A camera that drops for ten seconds during a five-hour
    session must cost ten seconds, not the remaining four hours - unattended
    is the whole point, and an RTSP session on these cameras does drop.
    """

    def __init__(self, name: str, url: str, out: Path, segment_s: int,
                 ffmpeg: str = "ffmpeg", tz_name: str = ""):
        self.name, self.url, self.out, self.segment_s = name, url, out, segment_s
        self.ffmpeg = ffmpeg
        # `-strftime` names segments from the system clock, and the GPU server
        # runs UTC while the app - and everyone reading the recordings - works
        # in Asia/Tashkent. Left alone, the same five hours would be named
        # 04:00 on the server and 09:00 on a laptop, and cross-camera alignment
        # by filename would silently be five hours out.
        self.env = dict(os.environ)
        if tz_name:
            self.env["TZ"] = tz_name
        out.mkdir(parents=True, exist_ok=True)
        self.proc: subprocess.Popen | None = None
        self.restarts = -1          # the first start is not a restart
        self.last_start = 0.0
        self._stderr: list[str] = []
        self._tail: threading.Thread | None = None

    def start(self):
        cmd = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",        # UDP drops packets at 4K
            "-timeout", "10000000",          # socket I/O timeout, microseconds
            "-i", self.url,
            "-an", "-c", "copy",             # no audio, no re-encode
            "-f", "segment",
            "-segment_format", "mp4",
            "-segment_time", str(self.segment_s),
            "-reset_timestamps", "1",
            "-strftime", "1",
            str(self.out / "%Y%m%d_%H%M%S.mp4"),
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE, env=self.env)
        self.restarts += 1
        self.last_start = time.time()
        self._tail = threading.Thread(target=self._drain, args=(self.proc,),
                                      daemon=True)
        self._tail.start()

    def _drain(self, proc: subprocess.Popen):
        """Keep ffmpeg's stderr moving.

        A full pipe buffer blocks the writer, which would wedge the recorder
        in a way that looks exactly like a dead camera.
        """
        for raw in iter(proc.stderr.readline, b""):
            line = _redact(raw.decode("utf-8", "replace").strip())
            if line:
                self._stderr.append(line)
                del self._stderr[:-20]
                log.debug("[%s] %s", self.name, line)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def last_error(self) -> str:
        return self._stderr[-1] if self._stderr else ""

    def segments(self) -> list[Path]:
        return sorted(self.out.glob("*.mp4"))

    def bytes_written(self) -> int:
        total = 0
        for p in self.segments():
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def stop(self):
        """SIGINT, not kill: ffmpeg finalises the open mp4 on the way out.

        A hard kill leaves the in-progress segment without its moov atom,
        which makes it unplayable - the last few minutes of the session, lost
        for no reason.
        """
        if self.alive():
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                log.warning("[%s] would not finalise, killing", self.name)
                self.proc.kill()
                self.proc.wait(timeout=5)


def gaps(segs: list[Path], segment_s: int) -> list[tuple[datetime, float]]:
    """Where the recording is not actually continuous.

    Reported because a test set with a silent hole in it is worse than a short
    one: you would go looking for a pass that was never recorded and conclude
    the pipeline dropped it.
    """
    stamps = []
    for p in segs:
        try:
            stamps.append(datetime.strptime(p.stem, "%Y%m%d_%H%M%S"))
        except ValueError:
            continue
    out = []
    for a, b in zip(stamps, stamps[1:]):
        missing = (b - a).total_seconds() - segment_s
        if missing > segment_s * 0.5:
            out.append((a + timedelta(seconds=segment_s), missing))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--minutes", type=float, default=None,
                    help="alternative to --hours, for short trial runs")
    ap.add_argument("--camera", action="append", default=None,
                    help="camera name; repeatable. Default: every enabled camera")
    ap.add_argument("--out", default=None,
                    help="default data/sessions/<start stamp>")
    ap.add_argument("--segment-minutes", type=float, default=5.0,
                    help="length of each file; smaller loses less to a crash")
    ap.add_argument("--min-free-gb", type=float, default=15.0,
                    help="stop cleanly rather than fill the filesystem")
    ap.add_argument("--ffmpeg", default="ffmpeg",
                    help="path to the ffmpeg binary; the GPU server has no "
                         "system ffmpeg and uses a static build")
    ap.add_argument("--force", action="store_true",
                    help="start even when the projection says it will not fit")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan and the space arithmetic, record nothing")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    if args.hours is None and args.minutes is None:
        args.hours = 5.0
    duration_s = (args.hours or 0) * 3600 + (args.minutes or 0) * 60
    if duration_s <= 0:
        raise SystemExit("  duration must be positive")
    segment_s = max(10, int(args.segment_minutes * 60))

    from app.config import settings

    # Fail here, not in the restart loop. A missing binary makes every ffmpeg
    # exit instantly, and the supervisor would cheerfully respawn it for five
    # hours and report a session of empty directories.
    if shutil.which(args.ffmpeg) is None and not Path(args.ffmpeg).is_file():
        raise SystemExit(f"  no ffmpeg at {args.ffmpeg!r}\n"
                         f"  pass --ffmpeg /path/to/ffmpeg")

    db = ROOT / "data" / "ematsy.db"
    if not db.exists():
        raise SystemExit(f"  no camera database at {db}")
    rows = sqlite3.connect(f"file:{db}?mode=ro", uri=True).execute(
        "select name, rtsp_url from camera where enabled=1 order by id").fetchall()
    if args.camera:
        want = {c.lower() for c in args.camera}
        rows = [r for r in rows if r[0].lower() in want]
    if not rows:
        raise SystemExit("  no matching enabled cameras")

    started = datetime.now(settings.tz)
    out_root = Path(args.out) if args.out else \
        settings.data_dir / "sessions" / f"{started:%Y%m%d_%H%M%S}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Space arithmetic up front. These cameras measure ~5 Mbps (H.264) and
    # ~8 Mbps (H.265) at 4K/20 fps, but bitrate is variable and a busy day
    # is a HIGHER bitrate, not the same one - which is exactly the day this
    # gets run. 12 Mbps per camera is the pessimistic figure.
    free_gb = shutil.disk_usage(out_root).free / GIB
    projected_gb = len(rows) * 12e6 / 8 * duration_s / GIB
    need_gb = projected_gb + args.min_free_gb

    print(f"\n  continuous 4K, stream copy - no re-encode, no frame buffer")
    print(f"  cameras   : {', '.join(r[0] for r in rows)}")
    print(f"  for       : {duration_s / 3600:.2f} h  "
          f"(until ~{(started + timedelta(seconds=duration_s)):%H:%M})")
    print(f"  segments  : {segment_s / 60:.0f} min per file" if segment_s >= 60
          else f"  segments  : {segment_s} s per file")
    print(f"  output    : {out_root}")
    print(f"  projected : ~{projected_gb:.0f} GB worst case, "
          f"{free_gb:.0f} GB free, floor {args.min_free_gb:.0f} GB")

    if free_gb < need_gb:
        msg = (f"  need ~{need_gb:.0f} GB (projection + floor), "
               f"have {free_gb:.0f} GB")
        if not args.force:
            print(f"\n  REFUSING TO START.\n{msg}\n"
                  f"  Point --out at a bigger filesystem, shorten --hours, or "
                  f"pass --force\n  to record until the floor stops it.\n")
            return 1
        print(f"  {msg} - continuing anyway (--force); "
              f"the floor will stop it early\n")
    else:
        print()

    if args.dry_run:
        print("  dry run, nothing recorded\n")
        return 0

    cams = [Recorder(name, url, out_root / name, segment_s,
                     ffmpeg=args.ffmpeg, tz_name=str(settings.tz))
            for name, url in rows]
    for c in cams:
        c.start()
        log.info("[%s] recording", c.name)

    deadline = time.time() + duration_s
    stop = threading.Event()

    def _sig(*_):
        print("\n  stopping, finalising the open segments...")
        stop.set()
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    ended_early = ""
    next_report = time.time() + 300
    try:
        while not stop.is_set() and time.time() < deadline:
            time.sleep(2.0)
            now = time.time()

            free_gb = shutil.disk_usage(out_root).free / GIB
            if free_gb < args.min_free_gb:
                ended_early = f"only {free_gb:.1f} GB free"
                log.warning("%s - stopping", ended_early)
                break

            for c in cams:
                if not c.alive() and now - c.last_start > RESTART_BACKOFF_S:
                    log.warning("[%s] ffmpeg exited (%s), restarting",
                                c.name, c.last_error() or "no message")
                    c.start()

            if now >= next_report:
                next_report = now + 300
                written = sum(c.bytes_written() for c in cams)
                elapsed = duration_s - (deadline - now)
                rate = written / max(elapsed, 1)
                projection = rate * duration_s / GIB
                log.info("%.0f min in - %.1f GB written, ~%.0f GB projected, "
                         "%.0f GB free", elapsed / 60, written / GIB,
                         projection, free_gb)
                if projection + args.min_free_gb > free_gb + written / GIB:
                    log.warning("at this bitrate the session will hit the "
                                "%.0f GB floor before it finishes",
                                args.min_free_gb)
    finally:
        for c in cams:
            c.stop()

    finished = datetime.now(settings.tz)
    manifest = {
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "requested_s": duration_s,
        "actual_s": round((finished - started).total_seconds(), 1),
        "ended_early": ended_early or None,
        "segment_s": segment_s,
        "timezone": str(settings.tz),
        "mode": "continuous stream copy (no re-encode)",
        "cameras": {},
    }

    print(f"\n  {'=' * 66}")
    total = 0
    for c in cams:
        segs = c.segments()
        size = c.bytes_written()
        total += size
        holes = gaps(segs, segment_s)
        lost = sum(s for _, s in holes)
        manifest["cameras"][c.name] = {
            "segments": len(segs), "bytes": size,
            "restarts": c.restarts, "gaps": len(holes),
            "seconds_missing": round(lost, 1),
            "last_error": c.last_error() or None,
        }
        print(f"  {c.name:10s} {len(segs):4d} files  {size / GIB:6.1f} GB  "
              f"{c.restarts} restart(s)  {len(holes)} gap(s)"
              + (f"  [{lost / 60:.1f} min missing]" if lost else ""))
        for at, missing in holes[:5]:
            print(f"             gap at {at:%H:%M:%S} - {missing:.0f}s")
        if len(holes) > 5:
            print(f"             ... and {len(holes) - 5} more")

    (out_root / "session.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  {total / GIB:.1f} GB in {out_root}")
    if ended_early:
        print(f"  ENDED EARLY: {ended_early}")
    print(f"  manifest   {out_root / 'session.json'}")
    print(f"  feed it to bench/eval_4k.py --dir {out_root}")
    print(f"  {'=' * 66}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
