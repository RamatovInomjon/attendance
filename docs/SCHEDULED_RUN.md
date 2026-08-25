# Scheduled capture run — 2026-08-25 07:00 Asia/Tashkent

## What starts, and how

A systemd **user** timer fires `scripts/scheduled_start.sh`, which waits for the
GPU, checks onnxruntime, applies camera config, then launches `scripts/run.py`.

```bash
systemctl --user list-timers ematsy-capture.timer   # confirm it is armed
journalctl --user -u ematsy-capture -f              # follow once it starts
systemctl --user stop ematsy-capture.service        # stop the run
systemctl --user disable --now ematsy-capture.timer # cancel the schedule
```

Linger is enabled (`loginctl enable-linger inomjon`), so the timer fires even
with nobody logged in.

**The GPU wait matters.** A model was training when this was scheduled. If the
card is still busy, onnxruntime does not error — it silently falls back to CPU
and runs ~15x slower, which looks like a performance problem rather than a
misconfiguration. The script therefore waits up to 60 minutes for 2.5 GB free
before starting, and logs loudly if it starts anyway.

## What gets saved

| what | where | notes |
|---|---|---|
| Video clips | `data/recordings/<camera>/` | event-triggered, 1920px, MP4 + JSON sidecar |
| Every frame of every track | `data/debug/<person>/frames/` | aligned + native crop per frame |
| Best shot per pass | `data/debug/<person>/` | frame, native crop, aligned, JSON |
| Unrecognized passes | `data/debug/_unknown/`, DB | with the identity each nearly matched |
| Attendance + events | `data/ematsy.db` | `recognition_event`, `daily_attendance` |
| Run log | `data/logs/run_<stamp>.log` | |

### Why clips are event-triggered

Continuous 4K across both cameras is ~14.4 GB/hour; a twelve-hour day would want
~173 GB against ~104 GB free, nearly all of it empty corridor. Recording only
while a track is alive — plus 2 s either side — keeps the person-passes, which
is the footage worth replaying. Measured: ~440 KB per 8-second clip at 1920px.

Each clip carries a JSON sidecar naming who the live system thought it saw, so a
replay can be scored against the original answer instead of judged by eye.

### Frame dump

`save_all_frames` writes **every gate-passing frame** of every track, tagged
`ok` or `gated` with its score. The best-shot capture answers "what did it
decide"; this answers "what did it have to choose from" — which is what
separates a threshold problem from an image-quality problem.

Both paths stop writing when free space falls below their floor
(`record_min_free_gb` 10 GB, `save_all_min_free_gb` 8 GB) rather than filling
the disk.

## Retention

```bash
python scripts/cleanup.py --dry-run      # show what would go
python scripts/cleanup.py                # recordings 30d, debug 14d, snapshots 90d
python scripts/cleanup.py --free-gb 20   # also trim oldest until 20 GB free
```

## Configuration in force

| setting | value |
|---|---|
| head detector | `yolov8n_head_960x544.onnx` (retrained, native 16:9) |
| recognizer | `adaface_ir101_finetune.onnx`, threshold **0.14** |
| aligner | DFA mobilenet, margin 1.30, sharp warp |
| cameras | 4K H.264, 20 fps, GOP 12, 16 Mbps |
| image | WDR off, 1/25, DNR 50, sharpness 50 (factory) |
| processing | every frame → ~19 fps per camera |

The **0.14 threshold is still provisional** — it is the base model's operating
point rescaled arithmetically, not a measurement. This run should produce enough
samples to set it properly.

## Training watchdog

`horus/crowdhuman_exp/train_watchdog.sh` watches the CrowdHuman run and, if it
dies before 100 epochs, resumes it at **batch 16**.

Ultralytics' `resume=True` reads its arguments back from the run's `args.yaml`
and ignores the command line, so the watchdog edits that file before relaunching
(keeping a timestamped backup). Passing `--batch 16 --resume` alone would
silently keep batch 32 and most likely die the same way.

It gives up after 5 restarts, and treats reaching 100 epochs or an early-stop as
a legitimate finish rather than a crash.

At ~342 s/epoch the run should finish around **01:38**, about five hours before
the capture starts.
