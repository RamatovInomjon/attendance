# Deploying the per-crop gallery floors

One commit. **Code plus one additive column** — `face_embedding.threshold`,
nullable, so every existing row keeps behaving exactly as it does today. No
model changes, no re-enrolment, no re-encryption.

`app/core/gallery.*.so` is tracked and travels with the pull; the server has no
sources for it. Nothing else moves.

---

## From the laptop

```bash
cd ~/projectAI/face_rec/face_recognition_airi
git push origin master
```

## On gpu6 (the iClaude terminal at https://aiscan.airi.uz/iclaude/)

```bash
cd ~/faceid/ematsy
git log --oneline -1              # note this — it is the rollback target
git pull origin master
./scripts/restart.sh
```

That is the whole deploy. `restart.sh` backs up the database, clears the stale
bytecode, runs the migration, pre-flights CUDA and the gallery, and refuses to
start unless exactly one process comes back on the right port. Read its output;
stop at the first line that is not what it says it should be.

Then, **only if corridor crops were already added** on this server before the
pull — they carry no floor yet, so they are still running on the old global
threshold:

```bash
~/faceid/venv/bin/python scripts/augment_gallery.py --recalibrate
```

It prints one line per stored crop and is a no-op when there are none.

---

## What to check afterwards

Open `https://aiscan.airi.uz/faceid/gallery/review` as an admin.

**It should no longer refuse everything.** The old message —

> two different people match at 0.208, at or above the 0.190 threshold …
> The pair: enrolment · Narmatov Abdukadir vs enrolment · Qo'shmatov Axmat

— named two **enrolment photographs**. That pair is in the gallery whatever you
select, so nothing could ever be added. It is now shown as a warning banner at
the top of the page, and it stops blocking.

Each crop now shows a `chegara` (floor) beside its score and margin. Most sit at
the global threshold; a few are higher, and those are the crops that sit close to
somebody else. Below its floor a crop names nobody, so adding one cannot create a
false accept against any face the system has measured.

Measured on the 29 candidates in the laptop's data: every added row ends up
exactly 0.02 above its own worst impostor, and no stored vector of another person
can reach any of them.

---

## The one decision this surfaces

`recognition_threshold_override=0.190` is set in the server's `.env`. The
calibrated value is **0.215**, and the Narmatov/Qo'shmatov enrolment pair scores
**0.208** — above the override. That is a live false-accept risk today, and it
has nothing to do with augmentation.

Two ways to close it, and it is your call which:

* `recognition_threshold_override=0.215` (or remove the line, which reaches the
  same value through the per-model calibration). Safer; recognises less.
* Re-photograph one of those two people and re-enrol. Keeps the lower threshold.

Leaving it as it is means those two can be recorded as each other.

---

## Rollback

Code only, and the column is additive — the previous build ignores it.

```bash
cd ~/faceid/ematsy
git checkout <the commit noted before the pull>
./scripts/restart.sh
```

Corridor crops added in the meantime stay in the gallery and revert to the global
threshold, which is what the old build did with them. To take them back out:
`python scripts/augment_gallery.py --revert --apply`.
