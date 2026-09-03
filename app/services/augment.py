"""Adding live corridor faces to the enrolment gallery, and taking them back out.

The gallery is enrolment photography: frontal, lit, close. The corridor is not.
The same people score ~0.85 against the gallery and 0.20-0.43 walking past, and
17 of 54 enrolled people were never recognised once in a full day. A face
recognised at 0.55 walking through the door is a better reference for next time
than any studio photograph.

WHY A HUMAN CHOOSES, AND WHY THE MACHINE STILL REFUSES
------------------------------------------------------
Adding a mis-recognised crop is not a small mistake. That face becomes a
permanent reference for the wrong person, so the same error gets easier next
time, and easier again - the gallery poisons itself, silently, and nothing in
the attendance data reveals it.

So there are two independent gates, and both must pass:

  * an ADMIN looks at every crop and decides. Only a person can say "that is
    not him", and that judgement is the whole point of the review page.
  * the code gives every added crop its OWN acceptance floor, measured against
    every other person's vectors we hold - enrolment photographs and other
    people's corridor crops alike - and set just above the worst of them. A
    human can be fooled by a bad crop of a lookalike; this makes being fooled
    survivable.

WHY A FLOOR PER CROP, AND NOT ONE GATE OVER THE WHOLE GALLERY
-------------------------------------------------------------
The gate used to be a single measurement: re-compute the gallery's worst
cross-identity pair with the selection included, and refuse if it reached the
recognition threshold. That is the right question asked the wrong way, and it
failed in both directions.

It refused things it had no business refusing. The worst pair in this gallery is
two ENROLMENT photographs - Narmatov Abdukadir against Qo'shmatov Axmat, 0.202
locally and 0.208 on the server, against a 0.190 threshold there. That pair is
already in the gallery and no selection can remove it, so the gate was
permanently shut, and its message blamed whichever crop the admin had chosen for
a defect that predated it. Augmentation was unusable on the one deployment that
needed it most.

And it was too blunt about what it did allow. One threshold for every row means
a crop is judged by a number calibrated on studio photography. Passing that gate
says the gallery as a whole stayed under the line - not that THIS crop is safe.

The floor answers the narrower question directly. Crop i is given
`max(global_threshold, worst_impostor_i + margin)`, so no vector we have ever
seen from another person can reach it: its false-accept rate against the
measured population is zero by construction, not by inheritance. And because
the floor is the LOWEST value with that property, the crop still fires at the
weakest similarity that is defensible - which is the entire point of adding it.
Rejection is per crop and names the person responsible, instead of refusing the
whole selection.

The gallery's own worst pair is still measured and still reported. It is a real
problem - two enrolled people who resemble each other above the threshold - but
it is not this feature's problem, and it must not block it.

This module is the single implementation of all of it.
`scripts/augment_gallery.py` and the web review page call in here, so the CLI
and the UI cannot enforce different rules.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)

TAG = "live:"          # marks an embedding that came from the corridor


@dataclass
class Candidate:
    """One recognised pass, offered as a reference for the person it named."""
    key: str                       # capture stem; unique within debug_dir
    employee_id: int
    name: str
    camera: str
    when: str
    score: float                   # what the pipeline scored at the time
    own: float = 0.0               # recomputed: similarity to its own person
    other: float = 0.0             # recomputed: to the nearest OTHER person
    margin: float = 0.0            # own - other; what makes it defensible
    frames: int = 0
    aligner: float = 0.0
    impostor: float = -1.0         # worst similarity to ANOTHER person's vectors
    impostor_name: str = ""        # who that was
    threshold: float = 0.0         # this crop's own acceptance floor
    vec: np.ndarray | None = field(default=None, repr=False)
    rejected: str = ""             # why it is not offered, if it is not


@dataclass
class ImpostorCheck:
    """What the selection would do, kept apart from what the gallery already is.

    `after` is the worst EFFECTIVE similarity between a chosen crop and another
    person - effective meaning after that crop's own floor is applied, which is
    the similarity at which it could actually name somebody. `before` is the
    gallery's worst existing pair, which the selection neither causes nor cures.
    Conflating the two is what made the old gate unusable.
    """
    before: float
    after: float
    threshold: float
    pair: tuple[str, str] | None = None            # responsible for `after`
    pre_existing: tuple[str, str] | None = None    # responsible for `before`
    unusable: list = field(default_factory=list)   # (name, when, needed floor)

    @property
    def safe(self) -> bool:
        return not self.unusable and self.after < self.threshold

    @property
    def gallery_unsafe(self) -> bool:
        """The enrolment set already holds a pair at or above the threshold.

        Reported, never blocking. Two enrolled people who resemble each other
        this much is a real false-accept risk, but it is one the enrolment
        photography created and only re-enrolment can fix.
        """
        return self.before >= self.threshold


def crop_path(key: str) -> Path | None:
    """The 112x112 the recognizer saw, for one capture key.

    Resolved and CONTAINMENT-CHECKED. `data/debug` is deliberately outside the
    public media mount, and a key arriving from a web request is caller input:
    without this, "../../.." reads any file the service can.
    """
    root = settings.debug_dir.resolve()
    for person in root.iterdir() if root.is_dir() else []:
        if not person.is_dir():
            continue
        p = (person / f"{key}_aligned.jpg").resolve()
        if p.is_file() and p.is_relative_to(root):
            return p
    return None


def _floor(impostor: float, thr: float) -> float:
    """The lowest acceptance floor at which `impostor` cannot get through."""
    return max(thr, float(impostor) + settings.augment_threshold_margin)


def calibrate(cands: list[Candidate], M: np.ndarray, owner: np.ndarray,
              names: dict[int, str] | None = None) -> None:
    """Give every candidate its own acceptance floor, in place.

    The impostor set is deliberately BOTH populations: the enrolment gallery
    `M`, and the other candidates. Studio photographs alone would understate it
    badly - the whole premise of this feature is that a corridor face and a
    studio face of the same person score 0.2 apart, so another person's
    corridor crop is by far the more informative probe, and it is the one a
    live query most resembles.

    A candidate that would need a floor above `augment_max_threshold` is
    rejected here. Not because it is dangerous - the floor makes it safe - but
    because it could never fire, and a row that never fires is worse than no
    row: it looks like coverage that does not exist.
    """
    live = [c for c in cands if c.vec is not None and not c.rejected]
    if not live:
        return
    thr = settings.threshold_for(settings.recognizer_model)
    names = names or {}

    V = np.stack([np.asarray(c.vec, dtype=np.float32) for c in live])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    emp = np.array([c.employee_id for c in live], dtype=np.int64)

    # -2.0 is below any cosine, so a person with no impostor anywhere - the
    # only-employee case - falls through to the global threshold.
    if len(M):
        G = np.where(np.asarray(owner)[None, :] != emp[:, None], V @ M.T, -2.0)
        g_at = G.argmax(axis=1)
        g_best = G[np.arange(len(live)), g_at]
    else:
        g_at = np.zeros(len(live), np.int64)
        g_best = np.full(len(live), -2.0, np.float32)

    L = np.where(emp[None, :] != emp[:, None], V @ V.T, -2.0)
    l_at = L.argmax(axis=1)
    l_best = L[np.arange(len(live)), l_at]

    for i, c in enumerate(live):
        if float(g_best[i]) >= float(l_best[i]):
            c.impostor = float(g_best[i])
            c.impostor_name = names.get(int(owner[int(g_at[i])]), "") if len(M) else ""
        else:
            c.impostor = float(l_best[i])
            c.impostor_name = live[int(l_at[i])].name
        c.threshold = _floor(c.impostor, thr)
        if c.threshold > settings.augment_max_threshold:
            who = f" ({c.impostor_name})" if c.impostor_name else ""
            c.rejected = (f"would need floor {c.threshold:.3f} to stay clear of "
                          f"another person{who} - too close to be usable")


def scan(min_score: float | None = None, min_margin: float = 0.10,
         min_frames: int = 8, min_aligner: float = 0.95,
         max_per_person: int = 5, max_similarity: float = 0.92,
         include_rejected: bool = False) -> list[Candidate]:
    """Every capture worth offering, best first, thinned per person.

    The margin is RECOMPUTED here against the live gallery rather than read
    from the capture file: everything written before the margin fix stores 0.0,
    and a filter on that would pass the entire history unexamined.
    """
    import cv2
    from app.core.geometry import to_normalized_chw
    from app.core.recognizer import FaceRecognizer
    from app.services.enrollment import load_gallery

    thr = settings.threshold_for(settings.recognizer_model)
    min_score = thr * 1.5 if min_score is None else min_score
    root = settings.debug_dir
    if not root.is_dir():
        return []

    gal = load_gallery()
    if len(gal) == 0:
        return []
    rec = FaceRecognizer(settings.model_path(settings.recognizer_model),
                         batch_size=settings.embed_batch)

    raw: list[Candidate] = []
    for meta_f in sorted(root.rglob("*.json")):
        try:
            m = json.loads(meta_f.read_text())
        except Exception:
            continue
        if m.get("employee_id") is None:
            continue
        if not meta_f.with_name(meta_f.stem + "_aligned.jpg").is_file():
            continue
        q = m.get("quality") or {}
        c = Candidate(
            key=meta_f.stem, employee_id=int(m["employee_id"]),
            name=m.get("name", ""), camera=m.get("camera", ""),
            when=str(m.get("timestamp_local", ""))[:19],
            score=float(m.get("score", 0.0)),
            frames=int(m.get("embedded_frames", 0)),
            aligner=float(q.get("aligner_score", 0.0)))
        if c.score < min_score:
            c.rejected = f"score {c.score:.3f} < {min_score:.3f}"
        elif c.frames < min_frames:
            c.rejected = f"only {c.frames} agreeing frames"
        elif c.aligner < min_aligner:
            c.rejected = f"aligner {c.aligner:.2f} < {min_aligner:.2f}"
        elif q.get("gate") not in (None, "PASS"):
            c.rejected = f"gate {q.get('gate')}"
        raw.append(c)

    by_person: dict[int, list[Candidate]] = {}
    for c in raw:
        by_person.setdefault(c.employee_id, []).append(c)

    out: list[Candidate] = []
    for emp, group in by_person.items():
        group.sort(key=lambda c: -c.score)
        kept: list[Candidate] = []
        mine = gal.owner == emp
        for c in group:
            if c.rejected:
                if include_rejected:
                    out.append(c)
                continue
            if len(kept) >= max_per_person:
                c.rejected = f"already have {max_per_person} for this person"
                if include_rejected:
                    out.append(c)
                continue
            p = crop_path(c.key)
            img = cv2.imread(str(p)) if p else None
            if img is None:
                continue
            # debug_capture upscales the recognizer's 112x112 to 224 for
            # viewing, so it has to come back down. Verified over 250 captures:
            # every one still matches its own person best.
            if img.shape[:2] != (112, 112):
                img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
            v = rec.embed(to_normalized_chw(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[None])[0]

            sims = gal.M @ v
            c.own = float(sims[mine].max()) if mine.any() else -1.0
            others = sims[~mine]
            c.other = float(others.max()) if others.size else -1.0
            c.margin = c.own - c.other
            if c.own <= c.other:
                c.rejected = "resembles another person more than its own label"
            elif c.margin < min_margin:
                c.rejected = f"margin {c.margin:.3f} < {min_margin:.2f}"
            elif any(float(v @ k.vec) > max_similarity for k in kept):
                c.rejected = "near-duplicate of one already offered"
            if c.rejected:
                if include_rejected:
                    out.append(c)
                continue
            c.vec = v
            kept.append(c)
        out += kept

    # One pass over everything that survived, so each crop's floor accounts for
    # every OTHER crop on offer and not just for the enrolment photographs.
    calibrate(out, gal.M, gal.owner, gal.names)
    if not include_rejected:
        out = [c for c in out if not c.rejected]
    out.sort(key=lambda c: (c.name, -c.margin))
    return out


def _gallery_rows():
    """Every stored embedding, normalised, with owners and display labels."""
    from sqlalchemy import select
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope

    with session_scope() as s:
        rows = s.execute(select(FaceEmbedding.id, FaceEmbedding.employee_id,
                                FaceEmbedding.vector,
                                FaceEmbedding.source_file)).all()
        names = dict(s.execute(select(Employee.id, Employee.full_name)).all())
    if not rows:
        return (np.zeros((0, 512), np.float32), np.zeros((0,), np.int64),
                [], [], names)
    M = np.stack([np.frombuffer(r[2], np.float32) for r in rows])
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    owner = np.array([r[1] for r in rows], dtype=np.int64)
    ids = [int(r[0]) for r in rows]
    tags = [str(r[3] or "") for r in rows]
    return M, owner, ids, tags, names


def _worst_pair(M, owner, labels):
    """The closest two rows belonging to DIFFERENT people."""
    if len(M) < 2:
        return -1.0, None
    sim = M @ M.T
    np.fill_diagonal(sim, -1)
    masked = np.where(owner[:, None] != owner[None, :], sim, -1)
    i, j = np.unravel_index(int(masked.argmax()), masked.shape)
    return float(masked[i, j]), (labels[i], labels[j])


def check_impostors(chosen: list[Candidate]) -> ImpostorCheck:
    """What the selection would actually do to false accepts.

    Two separate measurements, because they answer separate questions and the
    old code answered neither by merging them:

    `before` is the gallery's own worst cross-identity pair. On this gallery it
    is two ENROLMENT photographs, and it is above the deployed threshold. That
    is a genuine false-accept risk and it is surfaced - but no selection here
    causes it and none can cure it, so it does not refuse anything.

    `after` is the worst similarity between a chosen crop and a DIFFERENT
    person, measured after that crop's floor is applied. Since the floor is set
    just above exactly that similarity, this is below the threshold whenever
    calibration succeeded, and the check is really asking whether it did.
    """
    thr = settings.threshold_for(settings.recognizer_model)
    M, owner, _ids, _tags, names = _gallery_rows()
    labels = [f"enrolment · {names.get(int(e), e)}" for e in owner]
    before, pre = _worst_pair(M, owner, labels)

    if not chosen:
        return ImpostorCheck(before, -1.0, thr, None, pre)

    # Re-calibrate the selection against the gallery as it stands right now.
    # The floors carried on the candidates came from the scan, which may be
    # minutes old and may predate another admin's additions.
    calibrate(chosen, M, owner, names)

    unusable = [(c.name, c.when, c.threshold) for c in chosen if c.rejected]
    usable = [c for c in chosen if not c.rejected]
    after, pair = -1.0, None
    for c in usable:
        # What the crop can actually reach against another person once its own
        # floor is in force: the floor is expressed downstream as a shift, so
        # an impostor at `c.impostor` presents as `impostor - (floor - thr)`.
        effective = c.impostor - (c.threshold - thr)
        if effective > after:
            after = effective
            pair = (f"NEW · {c.name} ({c.when}, {c.camera})",
                    f"{c.impostor_name or 'another person'} at "
                    f"{c.impostor:.3f}, floored to {c.threshold:.3f}")
    return ImpostorCheck(before, after, thr, pair, pre, unusable)


def add(chosen: list[Candidate]) -> int:
    """Insert the chosen crops, each with its own floor, tagged for removal."""
    from app.db.models import FaceEmbedding
    from app.db.session import session_scope
    if not chosen:
        return 0
    thr = settings.threshold_for(settings.recognizer_model)
    with session_scope() as s:
        for c in chosen:
            s.add(FaceEmbedding(
                employee_id=c.employee_id, source_file=f"{TAG}{c.key}",
                vector=c.vec.astype(np.float32).tobytes(),
                dim=int(c.vec.shape[0]),
                model_name=settings.recognizer_model, quality=float(c.margin),
                threshold=float(c.threshold or thr)))
    # Every live row's floor is now potentially stale: the rows just inserted
    # are impostors for the rows already there. Recomputing the whole live set
    # is the only form of this that stays true as crops accumulate.
    recalibrate()
    log.info("gallery: added %d live embedding(s)", len(chosen))
    return len(chosen)


def recalibrate() -> int:
    """Recompute every live row's floor against the gallery as it now stands.

    Run after any change to the gallery. A floor is a statement about the rest
    of the gallery, so it stops being true the moment the gallery changes:
    adding crop B lowers nothing, but it means row A - calibrated before B
    existed - has an impostor it was never measured against. Removing a row can
    only lower floors, which is free accuracy, and this is what collects it.

    Returns the number of rows whose floor moved.
    """
    from sqlalchemy import update
    from app.db.models import FaceEmbedding
    from app.db.session import session_scope

    thr = settings.threshold_for(settings.recognizer_model)
    M, owner, ids, tags, names = _gallery_rows()
    live = [i for i, t in enumerate(tags) if t.startswith(TAG)]
    if not live or len(M) < 2:
        return 0

    idx = np.array(live, dtype=np.int64)
    # Same owner masked out, which also masks each row against itself.
    S = np.where(owner[None, :] != owner[idx][:, None], M[idx] @ M.T, -2.0)
    at = S.argmax(axis=1)
    imp = S[np.arange(len(idx)), at]

    moved = 0
    with session_scope() as s:
        for k, i in enumerate(live):
            floor = _floor(float(imp[k]), thr)
            s.execute(update(FaceEmbedding)
                      .where(FaceEmbedding.id == ids[i])
                      .values(threshold=floor))
            moved += 1
            if floor > settings.augment_max_threshold:
                log.warning(
                    "gallery: live row %d (%s) now needs floor %.3f to stay "
                    "clear of %s - it is safe but will effectively never "
                    "fire; consider removing it",
                    ids[i], names.get(int(owner[i]), owner[i]), floor,
                    names.get(int(owner[int(at[k])]), "another person"))
    return moved


def added() -> list[dict]:
    """Everything previously added from the corridor, newest first."""
    from sqlalchemy import select
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope
    with session_scope() as s:
        rows = s.execute(
            select(FaceEmbedding.id, FaceEmbedding.employee_id,
                   FaceEmbedding.source_file, FaceEmbedding.quality,
                   Employee.full_name, FaceEmbedding.threshold)
            .join(Employee, Employee.id == FaceEmbedding.employee_id)
            .where(FaceEmbedding.source_file.like(f"{TAG}%"))
            .order_by(FaceEmbedding.id.desc())).all()
    return [{"id": r[0], "employee_id": r[1], "key": r[2][len(TAG):],
             "margin": r[3] or 0.0, "name": r[4], "threshold": r[5] or 0.0,
             "dead": (r[5] or 0.0) > settings.augment_max_threshold}
            for r in rows]


def remove(ids: list[int]) -> int:
    """Delete live embeddings by id. Enrolment rows are never touched: the
    filter on the `live:` tag is what makes this safe to expose in a UI."""
    from sqlalchemy import delete
    from app.db.models import FaceEmbedding
    from app.db.session import session_scope
    if not ids:
        return 0
    with session_scope() as s:
        r = s.execute(delete(FaceEmbedding).where(
            FaceEmbedding.id.in_(ids),
            FaceEmbedding.source_file.like(f"{TAG}%")))
    n = int(r.rowcount)
    if n:
        # Removing a row can only LOWER the floors of the rows that remain, so
        # this is not housekeeping - it is recovering the recognition range
        # that the removed crop was costing everybody else.
        recalibrate()
    log.info("gallery: removed %d live embedding(s)", n)
    return n
