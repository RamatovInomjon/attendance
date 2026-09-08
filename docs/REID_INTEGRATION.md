# Integrating the research ReID work — what shipped, and what it is worth

**Date:** 2026-09-08
**Source:** the `integration/` research package from `phd/dissertatsiya2`
(model, standalone inference, regrouping prototype, measurements).

> **`integration/` is deliberately NOT in this repository.** One of its
> documents (`docs/KOZ_TEKSHIRUVI.md`) names employees beside physical
> descriptions of them, which is personal data about identifiable people in a
> public repo. Nothing in it is needed to run or deploy this: the model ships
> as `models/reid_osnet_x0_75_256x128_e512_fp16.onnx`, and every number the
> deployment relies on is reproduced here and in `app/config.py`. Citations to
> `integration/docs/...` in the code refer to that out-of-band package.

Four changes went in. They are independent: any one can be reverted from
`app/config.py` without touching the others.

| # | Change | Where | Off switch |
|---|---|---|---|
| 1 | ReID model swapped to OSNet-x0.75 | `reid_model` | point it back at the old file |
| 2 | Second-chance face match at **tracklet** level | `CameraWorker._second_chance` | `tracklet_face_threshold = 0` |
| 3 | Pseudo-person grouping of unregistered people | `app/services/pseudo_gallery.py` | `pseudo_person = false` |
| 4 | Face breaks a cross-camera tie the body could not | `ReidWorker._match` | `reid_match_face_tiebreak = false` |

## Validated on production, 2026-09-02 → 09-07

Six days of gpu6's own data: 10,391 person-passes, 1,018 named by the live path
(9.8%), 3,858 unknown rows, 68 incomplete attendance rows of 173.

| | baseline | this change | + gallery augmentation, matched FAR |
|---|---|---|---|
| named passes | 1,018 | 1,046 (+2.8%) | **1,054 (+3.5%)** |
| incomplete attendance rows | 68 | 65 | **61 (−10%)** |
| unknown rows an operator reviews | 3,858 | **~1,800 (−53%)** | same |
| cross-camera ReID links (281-pass corpus) | 15 | **29**, 0 wrong either way | same |
| false-accept rate of the naming rule | — | **0.78%** | 0.78% |

**Every threshold in this work was proposed, measured, and then left at its
original value.** That is the honest summary: the model swap and the grouping
feature are the gains; the threshold tuning was not.

---

## 1. The ReID model — strictly better, same operating point

The deployed `reid_resnet101_ibn_256x128_e2048_fp16.onnx` was exported from an
**E1-era checkpoint**, not the ResNet checkpoints it was thought to carry. The
research proved it three ways (ONNX metadata path, MD5, and a 0.31–0.47 cosine
between ONNX and PyTorch embeddings) — an export accident, not a bad model.

Re-measured **here**, on this corridor's own 281-pass corpus with this
repository's own tool, not on the research test set:

```
python bench/reid_match_eval.py --dir data/reid_passes --sweep \
       --model models/reid_osnet_x0_75_256x128_e512_fp16.onnx
```

| | R101-IBN (old) | OSNet-x0.75 (new) |
|---|---|---|
| file / compute | 85 MB · 6.52 GFLOPs | **2.7 MB · 0.60 GFLOPs** |
| embedding | 2048 | 512 |
| genuine median | 0.700 | **0.787** |
| impostor **max** | 0.585 | **0.512** |
| at 0.6769 + margin 0.05 | 15 correct, 0 wrong | **29 correct, 0 wrong** |

`reid_match_threshold` was re-derived and **landed on the same number**. The new
model widened the gap on both sides rather than shifting the scale, so recall
nearly doubled at an unchanged operating point and the threshold now sits 0.16
above the worst impostor instead of 0.09.

⚠ **Old vectors are in a different space.** `_match` and the pseudo gallery both
filter on `model_name`, so they never mix — but passes stored either side of the
swap cannot match each other. Change over on a day boundary.

### The export debt is paid

`scripts/export_reid.py` could not export this checkpoint: the vendored model
definition predated the training heads, so `head.*` came back "unexpected" and
the strict check refused it. The deployed file had to be exported out of band.
`scripts/vendor_reid/` is refreshed and the exporter now infers the head, so:

```
python scripts/export_reid.py integration/models/reid_osnet_x0_75_256x128_e512.pt
  osnet_x0_75  256x128  embed=512  epoch=130
  verify on 32 real crops: min cosine vs torch 0.999992
  verified
```

---

## 2. Second chance at a name — the attendance change

**The live rule throws away evidence it already has.** Every frame is matched on
its own, and then `vote_min_recognitions` (5) of them must agree. A pass that
yields three good frames names nobody however clearly each one scores. Those
embeddings were computed on the capture thread and discarded.

So `TrackState` now accumulates a running quality-weighted sum of every
embedding a track produced — `w = aligner_score × √ipd`, the research's own
formula, in a form that can be accumulated one frame at a time — and a pass that
ends unnamed is matched against the same gallery once more, at **0.30**.

**0.30 is stricter than the live 0.18/0.215.** That is the safety property: a
second chance at a *lower* bar would be a way to admit exactly what the live
threshold exists to refuse.

### Measured HERE, and it is not the research number

`integration/docs/HISOBOT_YUZ_TANA.md` §4 reports +23…30% more employee passes
per day at 97.4% precision. **That is a measurement of something else.** It was
made on stored 320-px body crops with the live quality gates *relaxed*. The
deployed rule combines only the frames that already **passed** those gates.

`bench/tracklet_recheck_eval.py` runs the real `CameraPipeline` over recorded
clips and applies the real rule. Over 191 clips of 2026-08-26 (every 4th, so the
sample spans the whole day), 66 completed passes:

```
python bench/tracklet_recheck_eval.py --dir data/recordings --stride 4 --sweep
```

| | result |
|---|---|
| named by the live vote | 28 of 66 |
| **agreement on those** | **22/22 (100%)**, 6 not re-called |
| recovered from the 38 unnamed | **2 (+7% attendance passes)** |
| disagreements, thresholds 0.20 → 0.40 | **0 at every point** |

**Read that honestly.** The precision evidence is good — zero disagreements
anywhere in the sweep — but *n* is small (22 judged passes, one day), and the
recovery is **+7%, not +23%**. The rule is safe and it is worth having; it is
not yet worth the research headline.

**The gap is the gates, and that is the next lever.** The offline experiment's
gain came from tracklet aggregation *plus* relaxed gating; only the first half
shipped. `min_aligner_score = 0.90` rejects frames before they are ever
embedded, so the template never sees them. Embedding gated frames would put more
work on the capture thread, which has a 50 ms budget and is the one thing ReID
work may never slow down — so it needs measuring (`bench/pipeline_fps.py`)
before it is attempted, not assuming.

Dropping the threshold to 0.20 raises recovery 2 → 5 with still zero
disagreements. It was **not** taken: 0.20 is below the live threshold, which
throws away the safety argument above for three passes on one day.

Every recovered pass is written with `recognition_event.source = "tracklet"`,
so threshold calibration can exclude it and an audit can tell the two rules
apart. In the arbiter a live commit outranks a tracklet recovery, so when both
cameras see one walk the live view carries the group's direction and snapshot.

---

## 3. Grouping the people who are not enrolled

Roughly half of all passes belong to somebody the face path cannot name, and
most of those are not employees. Every one was written as an independent
"unknown", so the record could say how many unrecognised *passes* happened and
not how many *people* — one visitor seen five times and five visitors seen once
were the same row count.

`app/services/pseudo_gallery.py` attaches each unnamed pass to a stable
pseudo-identity (`P-000123`), using **face and body together**:

```
face  ->  clothing-independent, works ACROSS days.   PRIMARY
body  ->  clothing-dependent, works within ONE day.  FALLBACK
```

Measured in the research over three days and ~4200 unnamed passes:

| mode | pair precision | pair recall | pseudo-people | count error |
|---|---|---|---|---|
| body alone | 91–94% | **3.5–7.6%** | 563/803/607 | 126–198% |
| face alone | 90–98% | 23–31% | 999/1240/1037 | 9–23% |
| **face + body** | **90–98%** | **29–32%** | **499/695/520** | **8–26%** |

Body alone regroups nobody — two strangers dressed alike out-score one person
seen twice. Its entire contribution is linking passes with **no usable face**,
and doing that halves the number of pseudo-people.

### The rules that keep it safe

* **A pseudo-person is named by FACE and the enrolment gallery. Never by body.**
  Of ~1100 unknown tracks a day only ~45 are a recoverable employee, and at that
  base rate a body rule writes more wrong attendance rows than right ones at
  *every* threshold measured (0.60 → 0.90).
* **Naming a group writes no attendance.** It relabels the group's history so an
  operator sees a name instead of a code. Those passes were observed without an
  identity; authoring transitions for them afterwards is the fabrication
  `app/services/corrections.py` exists to refuse.
* **A face under 20 px inter-pupil distance is discarded, not down-weighted** —
  it reaches 0.35 against strangers, so it is a source of wrong links.
* **Body evidence is confined to one business date and one ReID model.** Both
  failure modes are silent: yesterday's coat and another model's embedding space
  both produce perfectly plausible cosines.

### The honest ceiling

Recall is ~30%, so one person's passes still land in about three groups, and the
distinct-visitor count is accurate to roughly 15%. The cause is not the model
and not the tracker: **only 23–28% of passes contain a face with IPD ≥ 20 px**.
That is a camera-geometry ceiling. Mounting lower or closer would raise it more
than any model change — the research is explicit that this, not the network, is
the largest remaining reserve.

Good enough for counting visitors and dwell time. Nowhere near good enough for
attendance, which is why nothing here writes any.

### Cost

Matching runs against an **in-memory index**, not the table: one contiguous
matrix of face templates with a parallel owner array, rebuilt from the database
on first use and at each day rollover. The obvious per-pass `SELECT` would have
moved tens of megabytes of BLOB through the ORM every time somebody walked past.
Body templates are same-day by design, so they live in a small per-day dict.

### Schema

```sql
CREATE TABLE pseudo_person (...);          -- code, employee_id, K templates
ALTER TABLE reid_pass ADD COLUMN face_vector BLOB;      -- + face_dim/ipd/frames
ALTER TABLE reid_pass ADD COLUMN pseudo_person_id INT;  -- + pseudo_score/by
```

Applied by `python scripts/migrate.py` — additive, idempotent.

---

## 4. Face breaks a cross-camera tie the body could not

The runner-up margin in `ReidWorker._match` is expensive. Measured on this
corridor's 281-pass corpus, it drops correct matches from **55 to 29** while the
wrong count stays at **zero either way**. Those are not impostors being caught —
they are passes where two people's *clothing* scored alike and the body axis had
nothing left to say.

So when the body is undecided, and only then, the face is asked. Among the
candidates that already clear `reid_match_threshold` on body, one must clear
`pseudo_face_threshold` on face and beat the runner-up's face by
`reid_match_margin`. Otherwise the matcher abstains, exactly as before.

**The calibrated accept decision is untouched** — every candidate still has to
clear the body bar — so this can only recover a match the margin threw away,
never admit a new kind of one.

### A blended score was tried first, and removed

The research fusion (global-z blend, `w = 0.6`, FNIR@10% 11.59 → 5.93 over three
days, p < 1e-38) was measured under the **enrolment protocol**, where the
decision is a threshold on the blend. `_match` is a pairwise link with a
runner-up margin, and a z-normalised blend is not on the scale
`reid_match_threshold` was calibrated against.

Implemented and then reverted, because of what it actually did: a flipped winner
almost never *also* clears the body margin, so blending converted accepts into
rejections instead of improving them — a stricter matcher wearing fusion's name.
The tie-break adds recall exactly where the body has run out of information and
leaves the measured false-accept rate measured.

Global z, not per-query Z-norm, remains the right choice **if** a blend is ever
revisited: the textbook trick *doubles* single-modality error here
(11.59 → 25.29), because an open-set threshold is global and Z-norm destroys
precisely the absolute distance such a threshold reads.

## What was deliberately NOT taken from `integration/`

* **More enrolment crops per pass.** Measured to saturate: 7 → 10 crops buys
  ~1 point of FNIR, beyond 10 is inside the seed noise, and at a strict
  operating point more crops make it **worse** (FNIR@1% 38.7 → 48.8 → 57.3) by
  blurring the average template across outfits.
* **Multiple templates per person in the ReID gallery.** One quality-weighted
  mean beat both alternatives (11.59 vs 14.17 vs 14.85).
* **The McByte tracker.** `integration/docs/MCBYTE_BAHO.md` recommends against
  it, with evidence.

---

## Where to look

| | |
|---|---|
| model swap + threshold | `app/config.py`, `models/README.md` |
| second chance | `app/services/worker.py::_second_chance`, `app/core/pipeline.py` (`face_sum`) |
| grouping | `app/services/pseudo_gallery.py`, `app/db/models.py::PseudoPerson` |
| tie-break | `app/services/reid_worker.py::_match` |
| measurement | `bench/tracklet_recheck_eval.py`, `bench/reid_match_eval.py` |
| tests | `tests/test_tracklet_recheck.py`, `tests/test_pseudo_gallery.py`, `tests/test_reid_face_tiebreak.py` |
| research | `integration/README.md`, `integration/docs/` |

---

## 7. False accepts — the measurement that changed three decisions

Three thresholds were lowered on recall evidence and all three were put back
once FAR was measured. The recall evidence was correct; it was read at the
wrong base rate.

### Leave-one-person-out is how impostors were obtained

Production has three `resolved_kind='visitor'` rows in its whole history, so
there are no labelled non-employees to test with. `bench/tracklet_far_eval.py`
manufactures them: take every labelled CCTV pass of person X, delete X entirely
from the gallery, and match. X is now genuinely unenrolled, so any name
returned is a true false accept, and the probe is a real pass through this
corridor.

    thr      FPIR    genuine recall      (902 probes, 49 people, 6 days)
    0.18    2.66%        83.9%
    0.215   1.22%        80.6%
    0.24    1.11%        78.0%
    0.27    0.89%        76.2%
    0.30    0.78%        74.5%     <- deployed

### The probe set is balanced; production is not

That table has one impostor per genuine probe. This corridor has about fifteen:
of 1609 unnamed passes with a usable face over six days, only ~100 are a
recoverable employee. The recall gain applies to the small pool and the FPIR
gain to the large one, so the trade inverts:

    from 0.30    extra genuine   extra false   genuine per false
     -> 0.27         +3.2           +1.8             1.8
     -> 0.24         +3.7           +5.3             0.7
     -> 0.215        +2.9           +7.1             0.4
     -> 0.18         +2.8          +30.2             0.1

Genuine recoveries saturate around 94 whatever the threshold. Below 0.27 a
lower bar buys only wrong attendance rows.

### Why the grouping thresholds went back too

They were calibrated on the stored 320-px body crops, where **23.5%** of passes
carry a usable face. Production builds face templates from the full 4K frame,
where **72%** do. Replaying real footage (`bench/replay_prod_day.py`) showed
body 0.60 costing eleven points of pair precision against body 0.75 at the same
recall - and that comparison is not circular, because the grouping truth is
built from faces only. In a face-poor corpus the body rule was carrying far
more of the grouping than it ever does in production, and its errors were
hidden.

### The lever that does work: give people a CCTV gallery row

270 of the 328 gallery rows are webcam/ID photographs. Measured over 371
labelled CCTV passes, a CCTV gallery row is worth **+0.208** of median
similarity (0.310 -> 0.518), and at the deployed threshold the split is stark:

    34 people WITH a CCTV row      79.7% of their passes re-match
    15 people with WEBCAM only     40.3%

`scripts/augment_gallery.py` closes that gap. At **matched false-accept rate**
(baseline 0.30 vs augmented 0.40, both 0.78% FPIR):

    people who already had a CCTV row   79.7% -> 77.8%   (-1.9)
    people with webcam photographs only 39.7% -> 68.1%   (+28.4)
    people recognised on <half their passes:  9 -> 3

The aggregate ROC barely moves and is not supposed to. What it removes is the
SYSTEMATIC disadvantage of being enrolled from a webcam photograph. An employee
recognised 9% of the time has a broken timesheet every day, which is a worse
failure than a slightly lower average.

⚠ Augmenting DOES raise FAR at an unchanged threshold (0.78% -> 1.77%), because
a CCTV query scores higher against everybody's CCTV rows, impostors included.
The per-crop floors bound it; they do not remove it. Pair `--apply` with
`tracklet_face_threshold = 0.40`.

---

## 8. Deployment runbook

Nothing here has run end to end against a live camera. The replay drives
`CameraPipeline` directly, so `_second_chance` -> `PassArbiter` ->
`AttendanceService.record(source="tracklet")` and the pseudo-gallery's writes on
the ReID thread are exercised only by unit tests. Deploy in stages and verify
each one.

**0. Before anything** - back up the database.

    ssh gpu6@10.10.0.72 'cd ~/faceid/ematsy && cp data/ematsy.db data/ematsy.pre-reid-$(date +%Y%m%d_%H%M).db'

**1. Schema.** Additive and idempotent: one table, seven columns, two indexes.

    python scripts/migrate.py

**2. Model only, at a DAY BOUNDARY.** `reid_pass.vector` rows written by the
old model are in a different space. `_match` and the pseudo gallery both filter
on `model_name` so they never mix, but passes either side of the swap cannot
match each other - change over when the corridor is empty.

    pseudo_person=false
    tracklet_face_threshold=0

Verify: `/api/health` names the OSNet model; `pipeline_errors` stays flat; frame
timings unchanged; `reid` stats show passes being written.

**3. Grouping.** Set `pseudo_person=true`, restart. Verify the unknown-review
page shows `P-xxxxxx` chips and that `reid.pseudo` counts rise. Expect roughly
half as many distinct groups as unknown passes.

**4. Second chance.** Set `tracklet_face_threshold=0.30`, restart. Watch
`passes.recovered` in `/api/health`, and audit the first day by eye:

    select * from recognition_event where source='tracklet' order by ts desc;

Expect ~3-5 a day on this corridor. If any name looks wrong, set the threshold
back to 0 - it is a single value and needs no rollback.

**5. Gallery augmentation** (biggest single attendance gain, separate change):

    python scripts/augment_gallery.py --review     # LOOK at every crop
    python scripts/augment_gallery.py --measure
    python scripts/augment_gallery.py --apply      # then set threshold 0.40
    # undo: python scripts/augment_gallery.py --revert --apply

### Rollback

Every stage is a config value except the schema, which is additive and harmless
to leave in place. `reid_model` back to the ResNet file restores the previous
behaviour completely.
