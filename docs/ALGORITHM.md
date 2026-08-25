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
          [11] K-of-N vote ....... 3 agreeing frames commit an identity
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
| `min_face_px` | 66 | below this there is not enough face to identify |
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

## 11. K-of-N voting

An identity is committed when **3 of the last 5** scored frames agree on the
same person. One good frame is not enough; a single lucky match on a blurred or
half-turned face is exactly how a false accept happens.

For attendance, a **false accept is far worse than a miss**: one person is
recorded as another and nothing in the data reveals it, whereas a miss costs
nothing because the person is seen again.

The track keeps two snapshots, deliberately different:

- the **highest-scoring** frame — for diagnosing why a match happened
- the **clearest** frame — what a human is shown

These are usually not the same frame, and showing the score-best one to a person
reviewing a decision is misleading.

## 12. Attendance

The decision is made when the **track ends**, not when the vote commits — by
then the whole trajectory is known, so identity and direction are combined once,
with full information.

- **Direction wins over camera role.** The role only says where the camera
  points; the trajectory says what the person did.
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
| vote | 3 of 5 |
| per-frame cost | ~10 ms median, ~18 ms p90 (20% of the 50 ms budget) |
| gallery | 268 vectors / 55 people, d′ 10.03, rank-1 100% |

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
