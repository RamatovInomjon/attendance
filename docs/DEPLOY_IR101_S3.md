# Deploying the S3 IR-101 recognizer (2026-09-16)

This swap changes **the model, the threshold, the gallery and one .env line**.
Every step below is there because skipping it produces plausible-looking wrong
attendance rather than an error. Read `restart.sh`'s output at each stage and
stop at the first line that is not what it says it should be.

What is in the commit:

* `recognizer_model` → `ir101S3v2s_sr10final_fp16.onnx`, threshold 0.28,
  corridor-crop floor 0.44 (`models/README.md` records the measurements).
* Three tables that store face vectors now record which recognizer wrote them,
  and every reader checks it. Migration adds the columns and stamps old rows.
* `recognition_threshold_override` now applies only when
  `recognition_threshold_override_model` names the recognizer it was tuned for.
* Browser enrolments are saved as photo folders, so `enroll.py` can rebuild them.
* The pipeline restarts a track's identity when its head box jumps to another
  person (`track_head_jump_iou`); `head_jumps` appears in `/api/health`.
* `app/core/pipeline.*.so` rebuilt; the old ResNet101 ReID weights unpublished.

---

## 1. From the laptop

**The licence is already verified — nothing to check on the server.** The model
is encrypted with a key derived from a licence's `key_material`, so an `.enc`
built here opens on gpu6 only if the server's licence carries the same material.
It does: `licence_server.key` ("AIRI aiscan gpu6", fingerprint `f1e32d3a...`)
and the laptop's `licence.key` ("AIRI HQ", `b385055d...`) both hash to
`80374828ec55c9f0`, and every runtime model round-trips under the server
licence:

```bash
python scripts/encrypt_models.py --verify --licence licence_server.key
#   OK   ir101S3v2s_sr10final_fp16.onnx: round-trip matches      <- and the other four
```

That command reads the files and writes nothing, so re-run it any time -
after reissuing a licence, or if a deploy ever fails on decryption. Should it
print FAIL, re-encrypt against the server licence before shipping:

```bash
python scripts/encrypt_models.py --licence licence_server.key
```

Then:

```bash
cd ~/projectAI/face_rec/face_recognition_airi
git push origin master
scp models/ir101S3v2s_sr10final_fp16.onnx.enc gpu6@10.10.0.72:~/faceid/ematsy/models/
```

Only the `.enc` goes to the server. Never copy the plain `.onnx`, and never copy
a licence key - both licences are gitignored and both stay on this laptop. The
pre-flight in step 2 loads the recognizer through the vault, so a key problem
still fails there - before the cameras go down - with "decryption failed. The
licence does not match these model files".

## 2. On gpu6

```bash
cd ~/faceid/ematsy
git log --oneline -1                      # note it: the rollback target
git pull origin master
ls -la models/ir101S3v2s_sr10final_fp16.onnx.enc    # ~124 MB, must exist

# .env: the old override was tuned for the previous model and would be
# IGNORED anyway (the log will say so every minute) - remove it.
sed -i '/^recognition_threshold_override=/d' .env

./scripts/restart.sh
```

**This first restart is expected to refuse to start.** The migration runs and
stamps the old sightings, then the pre-flight finds a gallery built by the
previous recognizer and prints:

```
  recognizer  ir101S3v2s_sr10final_fp16.onnx
  GALLERY     gallery/recognizer mismatch: ...
  Rebuild it with THIS recognizer, then restart:
```

Do that:

```bash
~/faceid/venv/bin/python scripts/enroll.py
```

It re-embeds every enrolment photograph with the new model and **drops the
corridor crops** (they were the previous model's vectors and cannot be carried
over). At the end it prints one of two things:

* `gallery loaded: N embeddings / 57 people` — done; or
* `LEFT WITHOUT A FACE - ...` followed by names. These are people enrolled
  through the browser before captures were saved to disk; on gpu6 that is
  **To'raqulov Jaxongir, Kobilov Ilhomjon, Eshonqulova Feruza**. They must be
  re-enrolled once through *Ro'yxatdan o'tkazish* — this time the photos are
  kept, so it never has to be done again.

Then:

```bash
./scripts/restart.sh
```

The pre-flight should now read:

```
  recognizer  ir101S3v2s_sr10final_fp16.onnx
  gallery     ~268 embeddings / 54-57 people
  floors      0 corridor crop(s) with their own threshold (live floor 0.440)
  threshold   0.280
```

No `override` line. If one appears saying IGNORED, the sed above did not run.

## 3. What to check afterwards

* `/faceid/api/health` — `recognizer` names the S3 model; `head_jumps` is
  present per camera and stays small (a few per hour, not per minute).
* Watch two or three real passes on *Hodisalar*. Scores will read higher than
  before — a typical genuine match is now ~0.42 rather than ~0.34. That is the
  scale, not a change in confidence.
* *Galereya* is empty of corridor crops. Add them again over the next days as
  before; the 0.44 floor is applied automatically.
* After a week: `bench/far_live_rows.py`, and re-derive the floor if the
  measured naming rate says so.

## 4. Rollback

```bash
cd ~/faceid/ematsy
git checkout <previous commit>            # from step 2
./scripts/restart.sh                       # refuses: gallery is the new model's
~/faceid/venv/bin/python scripts/enroll.py # rebuilds it with the old recognizer
./scripts/restart.sh
```

`adaface_ir101_finetune_fp16.onnx.enc` is still on the server and still
listed in `recognizer_thresholds`, so nothing else is needed. The stamped
provenance columns stay and are harmless: old rows are simply not read by
whichever recognizer did not write them.
