"""Measure real decode cost for each stream option on both cameras."""
import time, cv2, os, sys
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer"
USER, PWD = "admin", "@a123456"
CAMS = {"in": "192.168.1.2", "out": "192.168.1.64"}

def bench(ip, ch, n=40):
    url = f"rtsp://{USER}:{PWD}@{ip}:554/Streaming/Channels/{ch}"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened(): return None
    for _ in range(5): cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    t0 = time.perf_counter(); ok = 0
    for _ in range(n):
        r, f = cap.read()
        if r: ok += 1
    el = time.perf_counter() - t0
    cap.release()
    return w, h, fps, ok, el / max(ok, 1) * 1000

for role, ip in CAMS.items():
    for ch in (101, 102):
        r = bench(ip, ch)
        if r is None: print(f"{role:4} ch{ch}: FAILED to open"); continue
        w, h, fps, ok, ms = r
        print(f"{role:4} ch{ch}: {w}x{h} @{fps:.0f}fps declared | {ok}/40 frames | {ms:6.1f} ms/frame ({1000/ms:5.1f} fps achieved)")
