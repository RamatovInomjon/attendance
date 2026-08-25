# Image quality investigation — 2026-08-19

"Frames are still too blurry." Four candidate causes, tested in turn.

## What it was not

**Optical focus is not the whole story, and H.264 is innocent.**

| test | result | verdict |
|---|---|---|
| sharpness across a 3×3 grid | peak regions 598 / 828 lapvar | lens resolves detail; low readings were featureless floor, where lapvar tracks contrast |
| ISAPI JPEG (no encoder) vs RTSP H.264 frame | H.264 retained **92–102%** of detail | compression is not the cause at 8 Mbps |

## What it was: temporal noise reduction

Measuring the **temporal** noise floor — per-pixel standard deviation across
consecutive frames of a static wall — gave the answer:

```
IN   temporal noise: 0.007 grey levels
OUT  temporal noise: 0.000 grey levels
```

A real sensor at maximum gain cannot produce a zero noise floor. The camera was
averaging across frames. On a static wall that looks immaculate; on a walking
face it is precisely a motion smear.

The chain that produced it:

```
shutter 1/250  ->  less light  ->  gain pushed to 100 (max)
               ->  heavy sensor noise
               ->  3D DNR at level 50 filters hard to hide it
               ->  temporal averaging  ->  moving faces smear
```

The 1/250 shutter set earlier in this project to fix motion blur is part of that
chain: it cut the light, which forced gain to maximum, which gave the noise
reduction more to do. It fixed one blur and fed another.

## Measured sweep

Static scene, 4K, same lighting throughout:

| shutter | DNR | sharpness | temporal noise | spatial detail | brightness |
|---|---|---|---|---|---|
| 1/250 | 50 | 50 | 0.012 | 180.7 | 114.8 |
| 1/250 | 15 | 50 | 0.551 | 428.3 | 116.2 |
| 1/250 | 0 | 50 | 0.600 | 431.9 | 116.3 |
| 1/125 | 15 | 50 | 0.343 | 472.8 | 120.9 |
| 1/250 | 10 | 55 | 0.603 | 482.6 | 114.9 |
| **1/250** | **10** | **60** | **~0.69** | **~520** | 115.0 |
| 1/250 | 5 | 60 | 0.863 | 547.3 | 114.9 |

Dropping DNR from 50 to 15 alone **more than doubles** spatial detail at
identical brightness. Nothing else in the sweep comes close.

`1/125` measures slightly better on a *static* scene — lower gain, less noise —
but a static test cannot see motion blur, and 1/125 doubles the smear on a
walking face. 1/250 is kept for that reason.

## Applied

| setting | was | now | effect |
|---|---|---|---|
| noise reduction | 50 | **10** | stops temporal averaging |
| sharpness | 50 | **60** | mild edge recovery |
| shutter | 1/250 | 1/250 | unchanged — freezes walking motion |

Result on both cameras:

| camera | detail before | detail after | temporal noise after |
|---|---|---|---|
| Entrance (IN) | 180.7 | **521.2** (2.9×) | 0.687 |
| Exit (OUT) | 318.9 | **833.8** (2.6×) | 1.272 |

Noise is back at a real sensor level rather than artificially zeroed, which is
what it should be: the recognizer tolerates noise far better than smear
(`bench/degrade_test.py` — 0.96 at sharpness 78, 0.71 at sharpness 47).

## Second finding: the Exit lens is soft

Edge-rise measurement (10–90% transition distance across the strongest edges,
contrast-normalised, so unlike lapvar it is not confounded by scene texture):

| camera | median edge rise | verdict |
|---|---|---|
| Entrance | 3.0 px | slightly soft |
| **Exit** | **4.0 px** | **needs refocusing** |

A focused 4K lens resolves an edge in ~1.5–2.5 px. `focusConfiguration` returns
`notSupport` on this model, so the focus ring is manual — this needs someone at
the camera. Loosen the lock screw, adjust while watching the live view, retighten.

Re-measure afterwards:

```bash
python - <<'EOF'
# see the edge_width() routine in this investigation
EOF
```

## The remaining limit is geometry, not settings

Live faces measure **97 px median** (p5 56, p95 201). Gallery enrolment photos
are 225 px. That gap is optics and mounting, and no camera setting closes it.

From the measured face size, the frame spans roughly 6.3 m at the distance
people are recognised — about 3 m out on a ~90° lens. To double face pixels you
must halve that span:

| option | effect | cost |
|---|---|---|
| **Lower the mount** to 2.0–2.2 m, aim more horizontally | biggest accuracy gain available — pitch is ArcFace's worst axis, and these cameras look steeply down | relocation only |
| **Narrower lens** (6 mm ≈ 54° instead of ~90°) or a varifocal model aimed at the doorway | ~2× face pixels | new camera or lens |
| **Dedicated face camera** at the choke point, wide cameras kept for overview | 200 px+ faces | one extra camera |
| **Light near the entrance** | lets gain drop, so DNR can stay low without noise | fitting |

Ranked by return per unit of effort: **refocus the Exit lens**, then **lower the
mounts**, then consider a narrower lens.

---

## Verification — third walkthrough, after the fix

Same person (enrolled), same corridor, same route.

| run | shutter | DNR | sharp | sharpness median | gate-pass | face px median | score median | score max |
|---|---|---|---|---|---|---|---|---|
| walk 1 | 1/25 | 50 | 50 | 36 | 12/116 (10.3%) | — | 0.142¹ | 0.226¹ |
| walk 2 | 1/250 | 50 | 50 | 32 | 85/200 (42.5%) | 97 | 0.214 | 0.514 |
| **walk 3** | **1/250** | **10** | **60** | **125** | **115/131 (87.8%)** | **126** | **0.254** | **0.496** |

¹ walk 1's subject was not enrolled, so its scores are impostor scores.

Effect of the DNR 50→10 and sharpness 50→60 change alone:

```
sharpness median    32 -> 125     3.9x
gate-pass rate    42.5% -> 87.8%  2.1x
score median      0.214 -> 0.254  +19%
observations >=0.30  27.6% -> 36.5%
```

**Nearly nine in ten observations now pass the quality gates**, against four in
ten before and one in ten at the start. The aligned crops confirm it visually:
eyes, nose and mouth texture are resolved where previously the face was a smear.
Visible grain is real sensor noise, which the recognizer tolerates well —
unlike the smear it replaced.

The identification was also clean: **61 observations, all matching one person**,
with no scattered runner-up identities of the kind the not-enrolled walker
produced.

## What still limits the score

Peak score is ~0.50 where gallery-to-gallery pairs reach 0.90. Image quality is
no longer the binding constraint; the crops show two remaining causes:

1. **Downward gaze.** The cameras look steeply down and people watch the floor
   while walking, so many frames are high-pitch views of a forehead. Pitch is
   ArcFace's weakest axis. Fixed by lowering the mounts, not by settings.
2. **Domain gap.** The gallery is clean 1280×720 enrolment photos.
   `scripts/enroll_from_live.py` adds confirmed corridor crops to close it.

Ranked by remaining value: **refocus the Exit lens** (4.0 px edge rise, free),
**lower the mounts** to 2.0–2.2 m, then **live enrolment**.

---

## Third round: testing an expert's recommended configuration

An external recommendation suggested: shutter 1/500 daytime, WDR **off** (or
20–40 dB), 3D DNR 15–25, sharpness 50–65, H.265 at 12–16 Mbps, I-frame 20.

Most of it matched what had already been measured here. Three points were
genuinely contested, so they were tested against real walking faces with
`scripts/ab_image_test.py`, which switches settings mid-session while somebody
keeps walking.

### A methodological error, and what it cost

The first two rounds optimised **Laplacian variance** as a sharpness proxy. That
was wrong: Laplacian variance rises with *noise* as readily as with detail, and
it cannot tell them apart.

At 1/500 the sensor is light-starved, gain pins at maximum, and with DNR low the
image is heavily grained. That configuration measured **3× "sharper"** than the
one it replaced — and scored **less than half** as well:

| run | config | n | sharpness median | **best score median** | best max |
|---|---|---|---|---|---|
| walk 3 | WDR50 · 1/250 · DNR10 · 8 Mb | 87 | 139 | **0.286** | 0.496 |
| walk 4 | WDR30 · 1/500 · DNR15 · 16 Mb | 9 | 754 | 0.323 | 0.535 |
| walk 5 | WDR30 · 1/500 · DNR15 · 16 Mb | 17 | 366 | **0.137** | 0.219 |

Walk 4 and walk 5 ran the **same configuration** and differed by more than 2× in
score — so walk 4's good result was the subject and their lighting, not the
settings. Against walk 3's much larger sample the 1/500 configuration loses
clearly.

The crops confirm it directly: blocky, speckled, and **dark** — mean brightness
105 against the enrolment gallery's 170. Dark, grainy faces are a domain shift
away from the gallery, whatever a gradient metric says about them.

**The only metric that counts is the recognition score.** Everything here is now
tuned on that.

### What the A/B run did establish

| config (faces 90–170 px, size held constant) | n | sharp med | best med | best max |
|---|---|---|---|---|
| WDR **off** · 1/500 | 16 | 213 | 0.150 | 0.221 |
| WDR **30** · 1/500 | 167 | 359 | **0.209** | **0.519** |

**Turning WDR off did not help** — the opposite of the recommendation. This
corridor has strong window backlighting; the exposure benefit of WDR outweighs
the frame-blending concern on these faces.

### Verdict on each recommendation

| recommendation | verdict |
|---|---|
| Shutter 1/500 daytime | **Rejected.** Starves the sensor here; faces come out dark and grainy and score worse. 1/250 retained. |
| Slow shutter OFF | N/A — exposure is manual-only on this model, there is no slow-shutter control. |
| WDR OFF | **Rejected on measurement.** WDR 30 beat WDR off at matched face sizes. WDR 50 retained. |
| BLC / HLC OFF | Already off. |
| 3D DNR 15–25 | **Agreed in principle**; 10 measured marginally better here and is retained. |
| Sharpness 50–65 | **Agreed** — 60 in use. |
| Higher bitrate (12–16 Mbps) | **Adopted.** 8192 → 16384. Costs nothing on a LAN and motion is what consumes bits. |
| H.265 instead of H.264 | **Not adopted.** H.265 is better per bit, but bitrate is not scarce here, and H.264 decodes more cheaply. Raising the bitrate captures the same quality. |
| I-frame interval 20 | **Not adopted.** GOP 12 (~1 s) recovers faster from a dropped connection. |

### Configuration in force

```
WDR         open, level 50
shutter     1/250
3D DNR      10
sharpness   60
codec       H.264, 3840x2160, 12 fps, GOP 12
bitrate     16384 kbps VBR
```

`scripts/camera_config.py --check` reports drift; `--apply` enforces it.

> **Order matters.** Writing any image/ISP setting silently resets the encoder
> block — framerate, GOP and bitrate revert to defaults while the write still
> returns `OK`. `camera_config.py` therefore applies image settings first,
> re-reads, then applies stream settings, and verifies everything afterwards.
> This was observed three separate times before it was understood.

### The remaining gap is still not exposure

Live faces sit around 105 mean brightness against the gallery's 170, and peak
scores plateau near 0.50. Neither is fixed by camera settings:

1. **Refocus the Exit lens** — 4.0 px edge rise, free.
2. **Lower the mounts** to 2.0–2.2 m — pitch is ArcFace's weakest axis.
3. **Live enrolment** (`scripts/enroll_from_live.py`) — adds corridor-domain
   crops to the gallery, which also absorbs the brightness difference.

---

## Correction — the tuning was wrong (A/B walkthroughs, 2026-08-20)

Three back-to-back 5-minute walkthroughs, same person, same route, same
lighting, settings changed between runs:

| run | shutter | DNR | WDR | n | gate-pass | struct/noise | score max | hits |
|---|---|---|---|---|---|---|---|---|
| **1 factory defaults** | 1/25 | 50 | off | 74 | 20.5% | **1.27** | **0.707** | **12** |
| 3 middle ground | 1/100 | 35 | off | 36 | 69.4% | 1.11 | 0.536 | 7 |
| 2 previous tuning | 1/250 | 10 | on | 20 | 80.0% | 0.79 | 0.256 | 1 |

**The factory defaults beat the tuned configuration outright.** Two errors led
to the wrong conclusion earlier.

### Error 1 — the metric could not tell sharpness from noise

Everything above was optimised against Laplacian variance, which sums squared
second derivatives. Sensor noise is high-frequency, so it inflates that number
exactly like real detail does. Dropping DNR from 50 to 10 doubled the noise
(1.26 → 2.49 grey levels) and the metric read it as **3.9× sharper**.

The corrected metric divides mid-frequency energy (facial structure) by
high-frequency energy (noise). It ranks the three runs 1.27 / 1.11 / 0.79 —
the same order as the recognition scores, which Laplacian variance did not.

```python
f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
mag = np.abs(f); H, W = gray.shape
Y, X = np.ogrid[:H, :W]
r = np.sqrt((Y - H//2)**2 + (X - W//2)**2)
struct_over_noise = mag[(r > 6) & (r < 22)].sum() / mag[r >= 22].sum()
```

### Error 2 — WDR was turned on and should not have been

WDR was enabled to stop the windows blowing out. It does that by compressing
dynamic range, which also **flattens contrast across the face** — the washed-out
pink cast visible in the phase-2 crops. Brightness went *up* (75.6 → 98.5) while
recognition collapsed. Backlit windows are a real problem, but WDR pays for them
in the one place that matters.

The advice to keep WDR off was correct and was dismissed too readily.

### What is actually limiting

The trade is light, not settings. The corridor is dim enough that any shutter
faster than about 1/50 forces gain to maximum, and the resulting noise costs
more than the motion blur it prevents:

- 1/25 → clean frames, but only 20.5% survive the quality gates (motion blur)
- 1/100 → 69.4% survive, noticeably noisier, peak score down a third
- 1/250 → 80% survive, but almost nothing is recognisable

More light at the entrance would break this trade — it lets gain fall, which
makes a faster shutter affordable. No camera setting can substitute for it.

### Standing configuration

Factory image settings (WDR off, 1/25, DNR 50, sharpness 50), with the stream
kept at H.264 4K 12 fps GOP 12 @ 16 Mbps — the encoder settings were never in
question and the higher bitrate costs nothing on a LAN.
