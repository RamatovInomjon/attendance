# Recognition pipeline — optimization plan

Target behaviour for the per-person recognition pass, and the tasks to get
there. Written to be executed by Claude Code against this repo.

**Status: DRAFT — awaiting approval. Do not implement any task below until the
plan is confirmed.**

---

## 1. The intended algorithm

Stated as the desired end state, independent of what exists today:

1. Track the **person** body. The detector has two classes, `head` (0) and
   `person` (1).
2. When a **head** is associated with that person **and the head is larger than
   a size threshold**, align it and extract features.
3. Keep tracking the person and **keep recognising for the whole time they are
   in view** — do not stop after the first few confirmations.
4. When the tracker reports the **last box** for that person (they have left the
   camera), finalise:
   - identity = the **majority** result across every recognition of that track;
   - the face shown = the **highest-scoring frame belonging to that majority
     identity**;
   - the dashboard image = a **crop of the person body**, not the aligned face.

So a pass with 30 recognitions yields one dashboard row: the identity most of
those 30 frames agreed on, illustrated by the best frame among the ones that
agreed, shown as a body crop.

---

## 2. What already exists (do not rebuild)

Verified in the current tree. These parts of the description are done:

| Requirement | Where | Notes |
|---|---|---|
| Person-body tracking | `settings.track_on = "person"` | ByteTrack runs on person boxes |
| Head↔person association | `pipeline.py` `_head_in`, `_offset_of`, `_synth_head` | Head centre containment, not IoU. A missed head is synthesised from the person box using the last observed offset, so the trajectory stays continuous |
| **Head-size gate before alignment** | `pipeline.py` ~495–521 | Pre-gates on `min_face_px` (66) **before** paying for the warp — this is already the optimisation described in step 2 |
| One decision per person-pass at track end | `worker.py::_persist_completed` ← `pipeline.py::_prune` | Attendance is already decided when the track completes, not at vote commit |
| Best frame scoped to the committed identity | `gallery.py::TrackVote.best_snapshot` | Fixed 2026-08-27 (`99cf420`). Before that, the displayed face could belong to a *different* person seen in the same track |
| `person_box` retained per track | `pipeline.py::TrackState.person_box` | Available; simply never captured as an image |

**Implication:** the plan is smaller than it first appears. Four changes, not a
rewrite.

---

## 3. What must change

### Gap A — recognition stops early
`pipeline.py:492`:
```python
if st.face_box is not None and (not st.vote.decided or self._tracing()):
```
Once `TrackVote` commits (3 of 5), the track stops being embedded unless the
debug tracing flag is on. Continuous recognition currently exists **only** as a
debug mode.

### Gap B — identity is first-past-the-post, not majority
`TrackVote` holds `votes: deque(maxlen=window)` (window 5) and commits as soon
as any identity reaches `required` (3) **within that rolling window**. It is
"first to 3 of the last 5", not "majority of all 30".

### Gap C — dashboard shows the aligned face, not the body
`CompletedTrack.crop` carries the 112×112 aligned face. There is no body crop
anywhere in the struct.

### Gap D — cost of always-on recognition
Long tracks exist: today's live data has passes of **162 s** and **592 s**.
At 20 fps that is 3 000–11 800 frames.

**Decision (owner, 2026-08-27): do not bound this. Accuracy and FAR=0 come
first — if a track yields 100+ embeddings, take all of them.** More evidence
per pass is the point, not a cost to be managed.

Current headroom for reference: ~8.9 ms of a 50 ms per-frame budget on the
server, 7 827 faces embedded per camera per ~11 h. Cost is to be *measured and
reported* after T1, not traded against accuracy.

---

## 4. Tasks

Each task is independently reviewable and independently revertible.

### T1 — Recognise for the whole track life
**Change:** make continuous recognition the default; remove the
`vote.decided` short-circuit at `pipeline.py:492`. Every gate-passing frame of
the track is embedded, for the whole time the person is in view. **No cap.**

**Keep:** the quality gates and the `min_face_px` pre-gate — those decide
*which* frames are worth embedding and must not be relaxed. They are what makes
the extra frames useful rather than noisy.

**Report, do not limit:** after deployment, record `faces_embedded` and
`timings.total` per camera from `/faceid/api/health` and note them here. If the
per-frame budget is ever genuinely exceeded the answer is more GPU or a smaller
detector — not fewer embeddings.

**Acceptance:**
- a track in view for 30 s yields on the order of hundreds of embeddings, not 3;
- `recognized_tracks` and `faces_embedded` from `scripts/benchmark.py` rise or
  hold — they must not fall;
- frame time is measured and reported (no pass/fail threshold).

**Files:** `app/core/pipeline.py`.

---

### T2 — Consensus identity over the whole track
**Change:** `TrackVote` decides by **consensus across all** observations of the
track, not first-to-K within a 5-frame window.

**The rule (owner's, 2026-08-27):** sort the recognition results; if the same
person accounts for **≥ `vote_consensus` (0.65)** of the *identified* frames,
that is the person — provided at least `vote_min_recognitions` (**5**) frames
agree on them.

The per-frame threshold is unchanged: `Gallery.match` already requires a frame
to clear `recognition_threshold` (0.18) and `second_best_margin` (0.045) before
it produces an identity at all, so every frame in the denominator is already
above threshold. **No additional aggregate threshold check is added** (owner,
2026-08-27).

**Design:**
- unbounded `Counter` of per-frame identities for the whole track;
- `provisional_id` — current leader, shown on the live overlay during the pass
  so the live view is not anonymous until the person leaves (owner confirmed,
  2026-08-27). Attendance ignores it and waits for `final_id`;
- `final_id` — consensus winner at track end, for the attendance decision;
- `best_score` / `best_margin` / `best_snapshot` stay scoped to the **final**
  identity. The per-identity structure added in `99cf420` already does this and
  must not be reverted to a global maximum;
- **misses are NOT counted in the denominator** (owner, 2026-08-27). Consensus
  is measured over the frames that produced an identity, not over every embedded
  frame. A track with 8 frames for A, 2 for B and 20 misses is **80% for A**,
  and commits A. Rationale given: 8 agreeing recognitions is ample evidence in
  its own right.
  *Residual exposure to measure, not to argue about now:* this is the case where
  a stranger whose frames mostly miss can look unanimous on the few that land.
  The `vote_min_recognitions = 5` floor is what guards it. Once live data
  exists, check the miss-ratio of confirmed false accepts against genuine
  passes — if impostors show a distinctly higher miss ratio, adding it as a
  secondary gate is cheap and would not disturb this rule.

**Floor (see §8):** a fraction alone cannot judge a short pass — 1 of 1 is 100%
consensus. A pass with fewer than `vote_min_recognitions` (**5**) recognitions
commits nobody and is recorded as an unknown sighting.

**Why this is the lever for FAR=0:** an impostor has no true identity in the
gallery, so their frames scatter across whoever is nearest by noise; a genuine
person's frames converge on one. Consensus measures exactly that, and it is a
different axis from score — which today's analysis showed is unusable on its
own (two confirmed impostors at 0.218 and 0.180 sit *inside* a genuine range
reaching down to 0.177). Suggestive evidence: the confirmed Xamdamov impostor
needed **21 embeddings** before three frames agreed, where genuine passes
typically converge within the first three or four.

**Acceptance:**
- a track of 30 observations, 21 for A and 9 scattered, commits **A**
  irrespective of arrival order;
- a track with 12 for A, 10 for B and 8 misses commits **nobody** (12/22 = 55%,
  under the 65% consensus bar; the 8 misses are excluded from the denominator);
- a track with 8 for A, 2 for B and 20 misses commits **A** (8/10 = 80%);
- a track with 4 for A and 0 for anyone else commits **nobody** (below the
  5-recognition floor) and is recorded as an unknown sighting;
- a track whose leader falls below the agreeing-frame floor commits **nobody**
  and is recorded as an unknown sighting;
- gallery d-prime and rank-1 unchanged; `recognized_tracks` not lower.

**Files:** `app/core/gallery.py`, `app/core/pipeline.py`, `app/config.py`, tests.

---

### T3 — Person-body crop for the dashboard
**Change:** capture a crop of the **person box** and carry it to the dashboard.

**Design:**
- new `TrackState.disp_person` populated from `st.person_box` **on the same
  frame that produced the best face for the leading identity**, so the body
  shown and the face that decided the identity are the same moment;
- new `CompletedTrack.person_crop`;
- `worker.py::_save_snapshot` writes it; the attendance row references it;
- the dashboard template renders the body crop, with the aligned face kept as
  the secondary/diagnostic image (do not delete it — it is what makes a wrong
  match explicable, per `tests/test_face_presence_gate.py`).

**Watch:** person boxes are tall (roughly 1:2.5). Fix the display box to a
sensible aspect and letterbox rather than distorting, or the dashboard grid
will look wrong.

**Acceptance:**
- a recognised pass shows a recognisable body crop containing the person;
- the aligned face is still reachable for diagnosis;
- no unprefixed media URLs (`media_path()` — see `test_ui_pages.py`).

**Files:** `app/core/pipeline.py`, `app/services/worker.py`,
`app/web/viewmodels.py`, `templates/`, tests.

---

### T4 — Finalise on the last box
**Change:** ensure the identity is finalised from the **final** majority at the
moment the tracker drops the person, and that this is the value written to
attendance.

Attendance is already decided at completion, so this is mostly wiring T2's
`final_id` into `CompletedTrack` and confirming no earlier path can write a
competing decision. Audit `pipeline.py:624` (`res.outcomes.append(...)` at vote
commit) against `worker.py::_persist_completed` and make explicit which one
owns the attendance write.

**Acceptance:** exactly one attendance decision per completed track; a track
that changes its leading identity mid-pass records only the final one.

**Files:** `app/core/pipeline.py`, `app/services/worker.py`.

---

## 5. Validation — mandatory for every task

The standing rule on this project is that the algorithm must never get worse.

```bash
python scripts/benchmark.py --out data/bench/before.json   # BEFORE the change
# ...implement...
python scripts/build_native.py            # the core is Cython-compiled
python scripts/benchmark.py --out data/bench/after.json
python scripts/benchmark.py --compare data/bench/before.json data/bench/after.json
python -m pytest tests/ -q
```

Two traps that have already cost time on this project:

- **A stale `.so` silently wins over edited `.py`.** Always rebuild before
  benchmarking, or you measure the old code.
- **Do not compare against an old stored baseline.** `data/bench/00_baseline.json`
  is only valid against the clip set it was recorded on; comparing across
  different clips produces a false "REGRESSION — do not ship". Always record a
  fresh `before.json` on the same clips.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Always-on recognition saturates the GPU during busy periods | T1's per-track cap and rate limit; re-measure `timings.total` on the live `/faceid/api/health` after deploy |
| Majority-over-all delays the live overlay (identity unknown until the pass ends) | Keep `provisional_id` for the live view; only attendance waits for the final |
| Body crops enlarge media storage | Body crops are larger than 112×112 faces — cap the stored width (~320 px) and monitor `media/` growth |
| Changing the vote rule shifts accuracy in ways the clip benchmark is too small to show (2 recognised tracks) | Additionally replay a day of live snapshots offline and compare identity decisions before/after |

---

## 7. Out of scope (tracked separately)

These are open defects found on 2026-08-27, not part of this plan, but they
touch the same code and should not be conflated with it:

- **Stale direction latch** — `st.direction` is latched while
  `st.direction_reason` refreshes, so a stationary person is committed with a
  direction up to 10 minutes old. Seven live instances observed.
- **Cross-camera duplicates** — one corridor transit recorded on both cameras
  as two contradictory transitions. A dozen instances observed, several with
  overlapping tracks.
- **`NO_CHECKIN` never clears** on a later check-in.
- **Gallery augmentation** — adding high-confidence live crops to the gallery
  measurably doubled margins for the augmented identity and left a known
  impostor unchanged. The strongest available lever against both the 42%
  miss rate and false accepts, but it needs its own reviewed-approval design.

---

## 8. Short passes — DECIDED

A consensus fraction needs a denominator big enough to mean something: **1 of 1
is 100%** and clears any percentage rule, so a single blurred or half-turned
frame could otherwise decide an identity with a perfect-looking consensus.

**Decision (owner, 2026-08-27): reject a pass with fewer than 5 recognitions.**
`vote_min_recognitions = 5`. Below that the pass commits nobody and is recorded
as an `UnknownSighting`, exactly as an unrecognised pass is today. No attendance
event is written.

This is the FAR=0-favouring choice: it discards some genuine quick walk-throughs
rather than risk an identity decided on too little evidence.

T1 shrinks the exposure on its own — once every frame is embedded instead of
bailing out at three votes, a 5-second walk at 20 fps yields far more than five
usable frames. What remains under the floor is only the genuinely brief or
badly-angled pass.

**Still measure the cost.** After T1+T2 land, replay a day of live passes and
count how many *genuine* recognitions the floor rejects. If that number is
material the floor can be revisited with evidence rather than intuition.

---

## 9. Suggested order

`T2 → T1 → T4 → T3`

T2 first because it is self-contained, unit-testable without the GPU, and
defines the `final_id` that T1 and T4 depend on. T1 second because its cost
profile is easier to judge once the vote rule is settled. T3 last because it is
the only one that touches the UI and it cannot break recognition.
