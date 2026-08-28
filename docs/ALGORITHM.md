# How the algorithm works

Two cameras watch one corridor. The system must answer, for every person who
walks through: **who was that, and were they coming in or going out?**

Everything below follows from one design decision: **track heads, recognise
faces.** A head stays visible for the whole pass; a face appears for only a
handful of frames. So the head carries the identity and the path, and the face
only has to be good *once*.

---

## The pipeline, end to end

```
RTSP 4K @20fps
      |
  [1] decode ..................... every frame, ~1.8 cores/camera
      |
  [2] downscale to 960 wide ...... 0.9 ms
      |
  [3] head detection ............. YOLOv8n @960x544, ~7-14 ms
      |
  [4] ByteTrack .................. head boxes -> stable track IDs
      |
      +--> [5] trajectory ........ where this person is going
      |
      +--> [6] size gate ......... skip faces too small to recognise
                |
           [7] alignment ......... DFA landmarks -> 112x112 warp
                |
           [8] quality gates ..... blur / pose / aligner score
                |
           [9] embedding ......... AdaFace IR-101 -> 512-d vector
                |
          [10] gallery match ..... cosine vs 268 enrolled vectors
                |
          [11] consensus ......... >=65% of identified frames, >=5 agreeing
                |
  [12] track ends -> combine identity + direction -> attendance row
```

Steps 1-4 run on **every** frame. Steps 6-11 run only for tracks that still
need an identity, and only on frames where the head is big enough.

---

## 1-2. Decode and downscale

Frames arrive at 3840x2160. Detection runs at 960 wide, but **crops for
recognition are always taken from the full-resolution frame** — downscaling is
for the detector's benefit only, and throwing away pixels before alignment would
be throwing away the face.

The reader thread keeps a **drop-oldest** queue of 4 frames. A backlog is
latency, not work: recognition wants the newest frame, so when the pipeline
falls behind it discards the stale frame rather than working through a queue.

## 3. Head detection

`yolov8n_head_960x544.onnx`, retrained on full CrowdHuman (mAP50 0.812), two
classes: `0=head, 1=person`. Only heads are used.

The export is **960x544, not square**. A 16:9 frame letterboxed into 960x960
spends 44% of every buffer on grey padding that the CPU converts and the GPU
convolves. Detection input buffers are preallocated once and reused, because
allocating a fresh 11 MB tensor per frame cost 6 ms of page faults on this host.

## 4. Tracking

Ultralytics `BYTETracker`, one instance per camera. Head boxes in, stable track
IDs out. One track ≈ one person-pass.

ByteTrack measures its lost-track buffer **in frames**, not seconds
(`max_time_lost = frame_rate/30 x track_buffer`). `track_frame_rate` must
therefore match the real processing rate or the occlusion tolerance silently
changes with it — at 20 fps and buffer 36 that is 24 frames, about 1.2 s.

## 5. Direction — the part that makes check-in/out correct

The naive design is "entrance camera = check-in". That is wrong here: both
cameras overlook the same corridor, so somebody *leaving* walks through the
entrance camera's view. Trusting the camera role writes a check-in for a person
walking out.

So direction is measured from the trajectory, using two independent signals:

| signal | what it is | strength |
|---|---|---|
| **tripwire crossing** | the head centre crosses a configured line; the sign of the cross product says which way | a geometric fact |
| **depth trend** | the head box grows as somebody approaches, shrinks as they leave | an inference |

The verdict:

- **both signals agree** → confident. This is the normal case.
- **crossing only** → trusted; a crossing is a fact.
- **depth only** → accepted, but requires more travel (0.15 vs the usual 0.06).
  Justified by observation: on every track where both signals fired they
  agreed, while 7 of 18 tracks crossed no line at all because people walk past
  the camera rather than through the middle of frame.
- **neither, or too few points** → `UNKNOWN`.

`UNKNOWN` is a deliberate refusal to guess, not a failure. A wrong direction
writes a wrong attendance row that nothing in the data reveals; a refusal costs
nothing, because the person is seen again on their next pass.

Direction needs `min_points=5` and real travel, which is why the pipeline
processes every frame rather than every second frame — at 10 fps a brisk walker
often failed to supply five points.

## 6. Size gate — before paying for alignment

`face_px` (the larger side of the head box) is known from the box alone, so
heads below `min_face_px` are dropped **before** alignment. Aligning one face
costs ~15 ms; this check costs nothing. It matters because the current detector
finds noticeably more distant heads, and each one used to be warped and scored
only to be discarded microseconds later.

## 7. Alignment

CVLFace **DFA-mobilenet** predicts five landmarks, which drive a Umeyama
similarity transform into a 112x112 crop.

Two subtleties:

- **`align_mode: sharp`.** The warp is sampled from a native-resolution crop,
  not from the 160x160 tensor the landmark model sees. Landmarks are cheap to
  compute at low resolution; pixels are not. Measured: d′ 12.63 vs 11.33.
- **`align_margin: 1.30`.** With a 160 px window, a margin of 1.30 puts about
  123 face pixels into a 112 px output — a slight downsample, which keeps
  detail without upsampling noise. Break-even is 1.43.

The aligner receives the **BGR frame directly** and converts only the crops it
reads. Converting a whole 4K frame to RGB cost 5.89 ms and a 24 MB allocation
to feed a couple of 200 px windows.

## 8. Quality gates

A face must pass all of these to be embedded:

| gate | value | why |
|---|---|---|
| `min_face_px` | 56 | below this there is not enough face to identify. Was 66, which rejected 5 575 of 7 578 head-bearing frames and cost recognitions for no accuracy gain |
| `min_laplacian_var` | 25 | motion blur destroys the texture AdaFace keys on |
| `min_aligner_score` | 0.40 | the aligner's own confidence in its landmarks |
| `max_yaw_deg` | 45 | profile faces embed poorly |
| `max_pitch_deg` | 40 | the binding constraint at current mount height |

Each frame also gets a **combined quality score** (size x sharpness x
frontality x aligner confidence) used to pick the best shot to show a human.

## 9-10. Embedding and matching

AdaFace **IR-101** produces a 512-d L2-normalised vector; matching is cosine
similarity against 268 enrolled vectors covering 55 people.

A match is accepted when it clears `recognition_threshold` **and** beats the
runner-up by `second_best_margin` (0.045). The margin matters: a face that is
nearly equally close to two people is not an identification.

> **The threshold belongs to the model.** 0.18 is calibrated for
> `adaface_ir101_finetune.onnx` and does not transfer. That model compresses
> impostor scores hard — gallery impostor max 0.208, against 0.392 for the base
> model it was fine-tuned from. Swapping the recognizer without re-calibrating
> would silently change what counts as a match.

## 11. Consensus voting

**Every gate-passing frame is recognised, for as long as the person is in
view**, and the identity is decided when the track ends — from all of it.

A person is committed when they hold:

* at least **65%** (`vote_consensus`) of the frames that produced *any*
  identity, **and**
* at least **5** (`vote_min_recognitions`) agreeing frames.

Otherwise the pass names nobody and is recorded as an unknown sighting.

**Misses are excluded from the denominator.** A frame matching nobody says
nothing about which of two candidates is right, so it does not dilute the
winner: 8 for A, 2 for B and 20 misses is 80% for A.

**The floor is what guards a short pass.** One agreeing frame out of one is
100% consensus and clears any percentage rule on its own — the fraction alone
cannot tell a confident pass from a lucky frame.

For attendance a **false accept is far worse than a miss**: one person is
recorded as another and nothing in the data reveals it, whereas a miss costs
nothing because the person is seen again. That asymmetry is why the floor
exists and why it is set where it is.

Consensus is also the strongest available defence against a false accept, and
it works on a different axis from score. An impostor has no true identity in
the gallery, so their frames scatter across whoever is nearest by noise; a real
person's converge. Score alone cannot separate them — two impostors confirmed
on 2026-08-27 scored 0.218 and 0.180, inside a genuine range reaching down to
0.177.

> **This replaced "first to 3 of the last 5".** That rule could settle an
> identity from the opening frames of a pass and never revisit it, and made the
> answer depend on arrival order. Live scores for one person spanned 0.196-0.473
> within a single day, so a pass's opening frames are not reliably its best
> evidence.

While the pass is running, `provisional_id` — the current leader — drives the
live overlay so the view is not anonymous until someone walks out of frame.
Attendance ignores it entirely and waits for the final consensus.

Every piece of evidence reported is **scoped to the identity being named**: the
score, the runner-up margin, the face and the body crop all come from that
person's own best frame. A track that holds two people cannot show one person's
name beside another's face — which it did, once, in production.

## 12. Attendance

The decision is made when the **track ends** — the only decision point. By then
the whole pass is known, so identity and direction are settled once, with full
information. `worker._persist_completed` is the only writer; the live feed
emitted during a pass carries provisional identities the consensus can still
overturn, and must never be persisted.

- **Direction wins over camera role.** The role only says where the camera
  points; the trajectory says what the person did.
- **A stale direction is no direction.** The verdict is timestamped when the
  trajectory supports it and discarded once older than `direction_max_age_s`
  (5 s) at the track's last sighting. Without this a person standing still kept
  a verdict up to ten minutes old, which booked check-outs on the entrance
  camera for people who had not moved.
- **Business day starts at 04:00** (Asia/Tashkent), so a late shift ending at
  01:00 files against the day it started.
- **First check-in of the day is kept**, not overwritten by later entries.
- An exit with no prior check-in records a check-out flagged `NO_CHECKIN`
  rather than being silently dropped.
- `event_cooldown_s` (90 s) suppresses duplicate rows when somebody lingers.

---

## Current parameters

| | value |
|---|---|
| stream | 4K H.264, 20 fps, RTSP over TCP |
| processed | every frame (~19 fps/camera, 95-97% of stream) |
| detection | `yolov8n_head_960x544.onnx`, conf 0.35, input 960 wide |
| tracker | ByteTrack, frame_rate 20, buffer 36 (~1.2 s tolerance) |
| aligner | DFA-mobilenet, 160 px window, margin 1.30, sharp warp |
| recognizer | AdaFace IR-101 fine-tune, threshold 0.18, margin 0.045 |
| vote | consensus: >=65% of identified frames, >=5 agreeing |
| direction latch | expires after 5 s without support |
| per-frame cost | ~10 ms median, ~18 ms p90 (20% of the 50 ms budget) |
| gallery | 268 vectors / 55 people, d′ 10.03, rank-1 100% |

## Debug capture

Clip recording and per-frame saving are **debug tools, off by default**. They
built the replay corpus that made the detector A/B, the direction comparison
and the fp16 validation measurable with the cameras switched off - but a
release keeps only one best-shot image per recognition, capped at 40 per
person, as attendance evidence.

## Where it can still fail

Honest list, in rough order of how often it bites:

1. **Pitch.** Cameras are mounted high enough that people are looked down on.
   Pitch is the binding quality constraint; lowering the mounts to 2.0-2.2 m is
   the highest-value physical fix available.
2. **Short tracks.** People who appear briefly, or in groups where boxes occlude
   each other, produce 1-2 point tracks that resolve no direction. Harmless for
   attendance (3 votes cannot be reached) but they inflate the "passes" count.
3. **Domain gap.** The gallery is enrolment photographs; the live domain is a
   corridor. Live scores (0.22-0.43) sit far below gallery scores (0.85 mean)
   for the same people. Enrolling from live crops would close this.
4. **The threshold is still provisional.** It rests on one confirmed false
   positive, not a labelled set.
5. **Direction near the frame edge.** Someone who enters and leaves on the same
   side never crosses the tripwire and may not travel far enough for a
   depth-only verdict.

---

## Where the frames actually go

Measured 2026-08-28 by replaying 20 recordings and counting the funnel. This is
the most useful single view of the pipeline, because it says where evidence is
lost rather than where time is spent.

| stage | frames | share |
|---|---|---|
| track alive | 8 247 | 100% |
| head box present | 7 578 | **92%** |
| **embedded** | **85** | **1%** |

Rejection reasons, and they are not close:

| reason | frames |
|---|---|
| **face too small** | **5 575** |
| aligner score | 418 |
| yaw | 54 |
| blur | 40 |
| pitch | 9 |

A head is found on almost every frame; the **size gate then discards nearly all
of them**. A 26-second pass embedded 12 of its ~520 frames. That single gate is
why a pass yields a handful of recognitions from ~196 frames, and it is the
first place to look for further recognition-rate gains — not the detector, not
the tracker, not the vote.

### Pass duration

From 107 live passes (2026-08-27) and replayed recordings, which agree:

| | |
|---|---|
| median | **9.8 s** |
| p25 / p75 | 7.2 s / 17.8 s |
| p90 | 76 s |
| max | 593 s |

Half of all passes are 5-10 s — a normal walk through the corridor. The
distribution is strongly right-skewed, so the mean (33 s) is misleading. The
13% running over 60 s are not slow walkers but **people standing still in
view**, and they are the ones that expose direction bugs: every stale-latch
instance came from a track of 76 s or longer.

Unrecognised tracks last *longer* than recognised ones (median 18.3 s vs 6.6 s).
Someone who walks briskly through presents a clean frontal face and is
recognised quickly; someone distant, turned away or loitering lingers without
ever offering a usable frame. Time in view is not the constraint — face quality
is.
