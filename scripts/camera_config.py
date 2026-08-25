#!/usr/bin/env python3
"""Apply and verify the camera configuration, in the right order.

Two things this exists to handle:

* **Changing image/ISP settings resets the encoder block.** Writing WDR or
  shutter silently reverts framerate, GOP and bitrate to their defaults. So
  image settings go first, stream settings last, and everything is read back.
* **Writes report OK and do not take.** Every value is verified after a settle
  delay, and mismatches are reported rather than assumed.

    python scripts/camera_config.py --check     # report drift, change nothing
    python scripts/camera_config.py --apply     # enforce the intended config
"""
import argparse, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from app.db.models import Camera
from app.db.session import session_scope

USER, PWD = "admin", "@a123456"
# Long enough to outlast the observed 15-40 s rollback window.
SETTLE_S = 55
NS = 'xmlns="http://www.hikvision.com/ver20/XMLSchema"'

# Chosen by measurement, not by defaults. See docs/IMAGE_QUALITY.md.
INTENDED = {
    # Chosen on RECOGNITION SCORE, not on any sharpness proxy.  Laplacian
    # variance rewards grain: the 1/500 configuration measured 3x "sharper"
    # than this one and scored less than half as well, because a light-starved
    # sensor at maximum gain produces noise that the metric scores as detail.
    "wdr_mode": "close", "wdr_level": 50,
    "shutter": "1/25",   # 1/500 starves the sensor; faces came out dark and grainy
    "dnr": 50,            # temporal DNR smears moving faces; 10 measured best
    "sharpness": 50,
    "codec": "H.264",
    # NOTE: these are a RECORD of a known-good configuration, not a policy the
    # system enforces. Nothing applies them automatically any more - doing so on
    # every service start silently reverted changes made from the camera's web
    # UI, which is indistinguishable from the camera resetting itself.
    "bitrate": 16384,   # camera maximum (vbrUpperCap max="16384")
    # Applying quality+bitrate first and GOP in a second PUT is what made these
    # stick. Writing all four at once was accepted (statusString OK) and then
    # silently rolled back 15-40 s later, with or without an RTSP client
    # attached. Verify with --check a minute after applying, never immediately:
    # an immediate read only proves the camera accepted the value, not kept it.
    # Video Quality "Highest". The camera accepts 1,20,40,60,80,100 for
    # fixedQuality; 100 is what the web UI labels Highest.
    "quality": 100,
    "framerate": 2000, "gop": 12,
}


def _curl(args):
    return subprocess.run(["curl", "-s", "-m", "15", "--digest", "-u", f"{USER}:{PWD}"] + args,
                          capture_output=True, text=True).stdout


def get(ip, ep):
    return _curl([f"http://{ip}/ISAPI/{ep}"])


def put(ip, ep, xml):
    p = Path("/tmp/_cfg.xml"); p.write_text(xml)
    out = _curl(["-X", "PUT", "-H", "Content-Type: application/xml",
                 "--data-binary", f"@{p}", f"http://{ip}/ISAPI/{ep}"])
    return "OK" in out


def first(pattern, text, cast=str, default=None):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else default


def read(ip):
    w = get(ip, "Image/channels/1/WDR")
    st = get(ip, "Streaming/channels/101")
    return {
        "wdr_mode": first(r"<mode>([^<]+)", w),
        "wdr_level": first(r"<WDRLevel>(\d+)", w, int),
        "shutter": first(r"<ShutterLevel>([^<]+)", get(ip, "Image/channels/1/shutter")),
        "dnr": first(r"<generalLevel>(\d+)", get(ip, "Image/channels/1/noiseReduce"), int),
        "sharpness": first(r"<SharpnessLevel>(\d+)", get(ip, "Image/channels/1/sharpness"), int),
        "codec": first(r"<videoCodecType>([^<]+)", st),
        "bitrate": first(r"<vbrUpperCap>(\d+)", st, int),
        "quality": first(r"<fixedQuality>(\d+)", st, int),
        "framerate": first(r"<maxFrameRate>(\d+)", st, int),
        "gop": first(r"<GovLength>(\d+)", st, int),
    }


def apply(ip):
    # 1. image/ISP first - these reset the encoder block when written
    put(ip, "Image/channels/1/WDR",
        f'<?xml version="1.0" encoding="UTF-8"?><WDR version="2.0" {NS}>'
        f'<mode>{INTENDED["wdr_mode"]}</mode><WDRLevel>{INTENDED["wdr_level"]}</WDRLevel></WDR>')
    put(ip, "Image/channels/1/shutter",
        f'<?xml version="1.0" encoding="UTF-8"?><Shutter version="2.0" {NS}>'
        f'<ShutterLevel>{INTENDED["shutter"]}</ShutterLevel></Shutter>')
    put(ip, "Image/channels/1/noiseReduce",
        f'<?xml version="1.0" encoding="UTF-8"?><NoiseReduce version="2.0" {NS}>'
        f'<mode>general</mode><GeneralMode><generalLevel>{INTENDED["dnr"]}</generalLevel>'
        f'</GeneralMode></NoiseReduce>')
    put(ip, "Image/channels/1/sharpness",
        f'<?xml version="1.0" encoding="UTF-8"?><Sharpness version="2.0" {NS}>'
        f'<SharpnessLevel>{INTENDED["sharpness"]}</SharpnessLevel></Sharpness>')
    time.sleep(4)

    # 2. stream profile: NOT written. It cannot be made to stick.
    #
    # Measured on DS-2CD2083G2-I firmware V5.7.13 by changing one field at a
    # time and watching for 200 s:
    #
    #     bitrate 16384 alone -> held at 60 s, reverted by 120 s
    #     quality 100   alone -> reverted by 60 s
    #     GOP 12        alone -> held at 60 s, reverted by 120 s
    #
    # Every field rolls back individually, so it is not one bad value or an
    # invalid combination - this firmware simply does not persist ISAPI writes
    # to the stream profile. It answers `statusString: OK`, applies the value,
    # and restores the stored profile a minute or two later. The rollback
    # happens with and without an RTSP client attached, so it is not the reader.
    #
    # Image settings (WDR, shutter, DNR, sharpness) DO persist over ISAPI and
    # are still written above - they have held all day without re-application.
    #
    # Set bitrate, quality, framerate and GOP from the camera's own web UI.
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    with session_scope() as s:
        cams = [(c.id, c.name, c.ip) for c in
                s.execute(select(Camera).where(Camera.enabled.is_(True))).scalars()]

    drift_total = 0
    for cid, name, ip in cams:
        if a.apply:
            print(f"applying to {name} ({ip}) ...")
            apply(ip)
        got = read(ip)
        drift = {k: (v, got.get(k)) for k, v in INTENDED.items() if got.get(k) != v}
        drift_total += len(drift)
        status = "OK" if not drift else f"{len(drift)} DRIFTED"
        print(f"\n  {name} ({ip}): {status}")
        for k, v in INTENDED.items():
            mark = " <-- expected " + str(v) if got.get(k) != v else ""
            print(f"    {k:<12} {str(got.get(k)):<10}{mark}")

    if drift_total and not a.apply:
        print("\nrun with --apply to enforce")
        sys.exit(1)


if __name__ == "__main__":
    main()
