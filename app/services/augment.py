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
    # `before` on the scale the MATCHER uses: each row's similarity less its own
    # floor. A corridor crop at 0.226 that only fires at 0.35 is not a risk, and
    # judging it by raw similarity reported a false accept that cannot happen.
    before_effective: float = -1.0

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
        return self.before_effective >= self.threshold


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


def corridor_probes(days: int = 14, dim: int | None = None):
    """Real corridor faces, for measuring what a crop can actually admit.

    `unknown_sighting.vector` is the face embedding of somebody the system named
    nobody. Hundreds accumulate every day at no cost, and they are the only
    population that resembles a live query: studio photographs do not, which is
    the entire premise of augmentation and was exactly the flaw in the first
    version of this check. Measured on gpu6, an enrolment photograph reaches a
    corridor crop at most 0.17, while a real corridor face reaches one at 0.31 -
    so calibrating against the gallery understated the danger by nearly double.

    Returns `(P, attributed)`: unit vectors, and the employee each probe is
    taken to BE, or -1. The caller uses it to drop probes that are the crop's
    own person seen again - see `calibrate`. An admin's label decides that
    where one exists (`_attributed`); otherwise it is the employee the
    pipeline found nearest when it rejected the face.
    """
    from datetime import date, timedelta

    from sqlalchemy import select
    from app.db.models import UnknownSighting
    from app.db.session import session_scope

    empty = (np.zeros((0, dim or 512), np.float32), np.zeros((0,), np.int64))
    cutoff = date.today() - timedelta(days=max(1, days))
    with session_scope() as s:
        rows = s.execute(
            select(UnknownSighting.vector, UnknownSighting.nearest_employee_id,
                   UnknownSighting.resolved_kind, UnknownSighting.resolved_employee_id)
            .where(UnknownSighting.vector.is_not(None),
                   UnknownSighting.business_date >= cutoff)).all()
    # A gallery rebuilt on another recognizer leaves older vectors of a
    # different width behind. Mixing them would not error - numpy would refuse
    # the stack, or worse, a same-width vector from another model would compare
    # as though it meant something. Keep only what matches the gallery.
    want = dim
    kept = [r for r in rows if r[0] is not None
            and (want is None or len(r[0]) == want * 4)]
    if not kept:
        return empty
    P = np.stack([np.frombuffer(r[0], np.float32) for r in kept])
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    near = np.array([_attributed(r) for r in kept], np.int64)
    return P, near


def _attributed(row) -> int:
    """Whose face a probe is, for the purpose of not counting as an impostor.

    A label from app/services/corrections.py beats the pipeline's guess. A
    sighting an admin resolved as employee X IS X, walking past unrecognised:
    a genuine probe for X's crops and an impostor for everyone else's, whatever
    the pipeline had found nearest. Judging it by `nearest_employee_id` alone
    counted X's own face against X unless the guess happened to agree. And a
    confirmed VISITOR is an impostor for everybody - including the employee
    the pipeline came closest to naming, which is exactly the lookalike the
    floor exists to keep out.
    """
    _vec, nearest, kind, resolved = row
    if kind == "visitor":
        return -1
    if kind == "employee" and resolved is not None:
        return int(resolved)
    return int(nearest) if nearest is not None else -1


def calibrate(cands: list[Candidate], M: np.ndarray, owner: np.ndarray,
              names: dict[int, str] | None = None, probes=None) -> None:
    """Set every candidate's acceptance floor, and refuse the lookalikes.

    THE FLOOR IS FLAT: `max(global_threshold, augment_live_floor)`. Every
    corridor crop answers to the same, higher bar, because that is what the
    measurement supports - see the table in app/config.py. A per-row floor from
    each crop's own worst impostor sounds better and measured worse on both
    axes: a maximum over a few hundred probes is a noisy extreme-value
    estimate, so it over-floors some rows while missing the lookalike who
    happened not to walk past on the day it was computed.

    What the per-row measurement is still for is REFUSAL. A crop that a real
    corridor face already reaches above the floor is a known lookalike; it is
    dropped and the collision is named, rather than being quietly given a floor
    of its own that hides the problem inside a number.

    The impostor set is three populations, worst wins:

      * the enrolment gallery `M` - weak, but free and occasionally decisive
      * the other candidates on offer
      * REAL CORRIDOR FACES from `corridor_probes()`, which dominate

    Probes whose `nearest_employee_id` is the candidate's own employee are
    dropped. Those are overwhelmingly the same person walking past again and
    being missed - precisely the case this feature exists to fix - and counting
    them as impostors would refuse a crop for resembling its own subject.
    """
    live = [c for c in cands if c.vec is not None and not c.rejected]
    if not live:
        return
    thr = settings.threshold_for(settings.recognizer_model)
    floor = max(thr, float(settings.augment_live_floor))
    names = names or {}

    V = np.stack([np.asarray(c.vec, dtype=np.float32) for c in live])
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    emp = np.array([c.employee_id for c in live], dtype=np.int64)

    # -2.0 is below any cosine, so a candidate with no impostor in a given
    # population - the only-employee case, or no probes yet - falls through it.
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

    if probes is None:
        probes = corridor_probes(dim=int(V.shape[1]))
    P, near = probes
    if len(P):
        S = np.where(np.asarray(near)[None, :] == emp[:, None], -2.0, V @ np.asarray(P).T)
        p_best = S.max(axis=1)
    else:
        p_best = np.full(len(live), -2.0, np.float32)

    for i, c in enumerate(live):
        c.threshold = floor
        options = [
            (float(g_best[i]), names.get(int(owner[int(g_at[i])]), "") if len(M) else ""),
            (float(l_best[i]), live[int(l_at[i])].name),
            (float(p_best[i]), "an unidentified corridor face"),
        ]
        c.impostor, c.impostor_name = max(options, key=lambda o: o[0])
        if _floor(c.impostor, thr) > floor:
            who = f" ({c.impostor_name})" if c.impostor_name else ""
            c.rejected = (
                f"another face already reaches this crop at {c.impostor:.3f}"
                f"{who}, at or above its {floor:.3f} floor - a lookalike, not a "
                f"reference")


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
    """Every stored embedding of an ACTIVE employee, normalised, with owners
    and display labels. The same population `load_gallery()` serves live:
    a deactivated person's rows can name nobody, so measuring floors and
    worst pairs against them reported risks the matcher could not take."""
    from sqlalchemy import select
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope

    with session_scope() as s:
        rows = s.execute(select(FaceEmbedding.id, FaceEmbedding.employee_id,
                                FaceEmbedding.vector, FaceEmbedding.source_file,
                                FaceEmbedding.threshold)
                         .join(Employee, Employee.id == FaceEmbedding.employee_id)
                         .where(Employee.is_active.is_(True))).all()
        names = dict(s.execute(select(Employee.id, Employee.full_name)).all())
    if not rows:
        return (np.zeros((0, 512), np.float32), np.zeros((0,), np.int64),
                [], [], names, np.zeros((0,), np.float32))
    M = np.stack([np.frombuffer(r[2], np.float32) for r in rows])
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    owner = np.array([r[1] for r in rows], dtype=np.int64)
    ids = [int(r[0]) for r in rows]
    tags = [str(r[3] or "") for r in rows]
    thr = settings.threshold_for(settings.recognizer_model)
    # The floor each row actually answers to. An enrolment photograph has none
    # and is judged at the global threshold; a corridor crop carries its own.
    floors = np.array([max(thr, float(r[4])) if r[4] else thr for r in rows],
                      dtype=np.float32)
    return M, owner, ids, tags, names, floors


def _source_label(tag: str, name) -> str:
    """What a row IS, not what it is assumed to be.

    This used to label every row "enrolment · <name>", including corridor
    crops. The worst pair on gpu6 was then reported as two enrolment
    photographs of Mahmudjon Azimov and Oybek O'ljaboyev at 0.226 - when it was
    in fact two CORRIDOR CROPS, which answer to a 0.35 floor and cannot name
    anybody at 0.226. The warning was wrong about what it was showing and wrong
    that there was anything to fix, and it sent somebody looking through
    enrolment photographs for a problem that was not in them.
    """
    return f"{'koridor kadri' if str(tag).startswith(TAG) else 'enrolment'} · {name}"


def _worst_pair(M, owner, labels, floors=None):
    """The closest two rows belonging to DIFFERENT people, AS THEY ARE JUDGED.

    Similarity alone is the wrong measure once rows answer to different floors.
    A corridor crop at 0.226 from another person's face is harmless if it only
    fires at 0.35; an enrolment photograph at 0.226 is a false accept at a 0.220
    threshold. So the comparison is made on the scale the matcher actually uses
    - each row's similarity less its own floor - and the raw similarity is
    reported alongside, because that is the number a human recognises.
    """
    if len(M) < 2:
        return -1.0, -1.0, None
    sim = M @ M.T
    np.fill_diagonal(sim, -1)
    masked = np.where(owner[:, None] != owner[None, :], sim, -1)
    if floors is None:
        i, j = np.unravel_index(int(masked.argmax()), masked.shape)
        return float(masked[i, j]), float(masked[i, j]), (labels[i], labels[j])
    # Row i names its own person, so it is row i's floor that must be cleared.
    # Expressed as a shift, exactly as Gallery._per_person applies it, so
    # "effective >= threshold" here means the same thing it means live.
    thr = settings.threshold_for(settings.recognizer_model)
    risk = masked - np.asarray(floors, np.float32)[:, None] + thr
    i, j = np.unravel_index(int(risk.argmax()), risk.shape)
    return float(masked[i, j]), float(risk[i, j]), (labels[i], labels[j])


def worst_pairs(limit: int = 5) -> list[dict]:
    """The closest pairs of enrolment photographs belonging to DIFFERENT people.

    Shown so they can be looked at. Two studio photographs that resemble each
    other above the recognition threshold are a live false-accept risk, and it
    is the one class of problem augmentation cannot help with: the crops answer
    to their own floor, but the enrolment set is judged at the global threshold
    and always has been. On gpu6 this pair reaches 0.226 against a 0.220
    threshold.

    Only pairs at or above the threshold are worth a person's attention, so
    that is the cut. Everything needed to render and act on them comes back
    with each row - both sides, their embedding ids, and the file each came
    from - because "your gallery has a problem" is not actionable and "these
    two photographs, this similarity" is.
    """
    thr = settings.threshold_for(settings.recognizer_model)
    M, owner, ids, tags, names, floors = _gallery_rows()
    if len(M) < 2:
        return []
    sim = M @ M.T
    np.fill_diagonal(sim, -1.0)
    masked = np.where(owner[:, None] != owner[None, :], sim, -1.0)
    # Judged the way the matcher judges: row i names its own person only once
    # row i's own floor is cleared. A corridor crop reaching another face at
    # 0.226 is harmless while it answers to 0.35, and reporting it as a problem
    # sends somebody hunting through photographs that are not the cause.
    risk = masked - np.asarray(floors, np.float32)[:, None] + thr

    out: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for k in np.argsort(risk, axis=None)[::-1]:
        i, j = np.unravel_index(int(k), risk.shape)
        if float(risk[i, j]) < thr or len(out) >= limit:
            break
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)

        def side(x):
            return {"id": ids[x], "employee_id": int(owner[x]),
                    "name": names.get(int(owner[x]), str(owner[x])),
                    "file": tags[x], "live": bool(tags[x].startswith(TAG)),
                    "floor": float(floors[x])}
        out.append({"similarity": float(masked[i, j]),
                    "effective": float(risk[i, j]), "threshold": thr,
                    "a": side(i), "b": side(j)})
    return out


def enrolment_image(embedding_id: int) -> Path | None:
    """The photograph one enrolment embedding was built from.

    Resolved and CONTAINMENT-CHECKED against `gallery_dir`, for the same reason
    `crop_path` is: the id arrives from a web request. A browser-captured
    enrolment has no file on disk and returns None.
    """
    from sqlalchemy import select
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope

    with session_scope() as s:
        row = s.execute(
            select(FaceEmbedding.source_file, Employee.folder)
            .join(Employee, Employee.id == FaceEmbedding.employee_id)
            .where(FaceEmbedding.id == int(embedding_id))).first()
    if not row or not row[0] or not row[1]:
        return None
    root = settings.gallery_dir.resolve()
    p = (root / row[1] / row[0]).resolve()
    return p if p.is_file() and p.is_relative_to(root) else None


def remove_enrolment(ids: list[int]) -> dict:
    """Delete enrolment embeddings, refusing to un-enrol anybody by accident.

    Separate from `remove()`, which is deliberately unable to touch anything
    but a `live:` row. This one can, so it carries the guard that one does not
    need: a person whose last embedding is deleted stops being recognisable
    with nothing in the UI to say why. That is refused.

    The photograph on disk is NOT deleted. `scripts/enroll.py` rebuilds the
    whole gallery from `face_id_users/`, so a row removed here comes back on
    the next enrolment run - which is the right default, because this is a
    judgement about one embedding and re-enrolment is a decision about the
    source data. The caller is told, so it can say so.
    """
    from sqlalchemy import delete, func, select
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope

    ids = [int(i) for i in ids]
    if not ids:
        return {"removed": 0, "refused": []}
    with session_scope() as s:
        rows = s.execute(
            select(FaceEmbedding.id, FaceEmbedding.employee_id,
                   FaceEmbedding.source_file, Employee.full_name)
            .join(Employee, Employee.id == FaceEmbedding.employee_id)
            .where(FaceEmbedding.id.in_(ids))).all()
        # ENROLMENT rows only. Counting a person's corridor crops here would
        # let their last studio photograph be deleted because they happen to
        # have two crops - leaving them matchable only through a 0.35 floor,
        # which is a quieter version of exactly the failure this refuses.
        counts = dict(s.execute(
            select(FaceEmbedding.employee_id, func.count())
            .where(FaceEmbedding.employee_id.in_([r[1] for r in rows]),
                   FaceEmbedding.source_file.not_like(f"{TAG}%"))
            .group_by(FaceEmbedding.employee_id)).all())

        ok, refused = [], []
        for eid, emp, src, name in rows:
            if str(src or "").startswith(TAG):
                refused.append(f"{name}: use the corridor-crop control for that one")
                continue
            left = counts.get(emp, 0) - sum(1 for x in ok if x[1] == emp) - 1
            if left < 1:
                refused.append(f"{name}: this is their last photograph - "
                               f"removing it would un-enrol them silently")
                continue
            ok.append((eid, emp))
        if ok:
            s.execute(delete(FaceEmbedding)
                      .where(FaceEmbedding.id.in_([x[0] for x in ok])))
    log.info("gallery: removed %d enrolment embedding(s), refused %d",
             len(ok), len(refused))
    return {"removed": len(ok), "refused": refused}


def check_impostors(chosen: list[Candidate]) -> ImpostorCheck:
    """What the selection would actually do to false accepts.

    Two separate measurements, because they answer separate questions and the
    old code answered neither by merging them:

    `before` is the gallery's own worst cross-identity pair. On this gallery it
    is two ENROLMENT photographs, and it is above the deployed threshold. That
    is a genuine false-accept risk and it is surfaced - but no selection here
    causes it and none can cure it, so it does not refuse anything.

    `after` is the worst similarity between a chosen crop and any other face -
    an enrolled person, another candidate, or a real corridor face - measured
    after the live floor is applied. A crop that a face already reaches above
    that floor is refused outright as a lookalike, so this reports how much
    headroom the accepted ones actually have.
    """
    thr = settings.threshold_for(settings.recognizer_model)
    M, owner, _ids, tags, names, floors = _gallery_rows()
    labels = [_source_label(t, names.get(int(e), e))
              for t, e in zip(tags, owner)]
    before, before_eff, pre = _worst_pair(M, owner, labels, floors)

    if not chosen:
        return ImpostorCheck(before, -1.0, thr, None, pre, [], before_eff)

    # Re-calibrate the selection against the gallery as it stands right now.
    # The floors carried on the candidates came from the scan, which may be
    # minutes old and may predate another admin's additions.
    calibrate(chosen, M, owner, names)

    # (who, when, the similarity another face already reaches it at) - the
    # measurement, not the floor, because the floor is now the same for all of
    # them and says nothing about why this one was refused.
    unusable = [(c.name, c.when, c.impostor) for c in chosen if c.rejected]
    usable = [c for c in chosen if not c.rejected]
    after, pair = -1.0, None
    for c in usable:
        # What another face can actually reach through this crop once its floor
        # is in force: the floor is expressed downstream as a shift, so a face
        # at `c.impostor` presents as `impostor - (floor - thr)`.
        effective = c.impostor - (c.threshold - thr)
        if effective > after:
            after = effective
            pair = (f"NEW · {c.name} ({c.when}, {c.camera})",
                    f"{c.impostor_name or 'another face'} at "
                    f"{c.impostor:.3f}, floored to {c.threshold:.3f}")
    return ImpostorCheck(before, after, thr, pair, pre, unusable, before_eff)


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
    """Put every stored live row on the current floor, and flag the lookalikes.

    Run after any change to the gallery, after a threshold change, and nightly.
    The floor itself is flat, so this is cheap - but two things do move under
    it. `recognition_threshold_override` can rise above `augment_live_floor`,
    in which case the floor must follow it up; and the corridor probe set grows
    every day, so a crop that looked clean in September can be shown to be a
    lookalike in October by a face that had simply not walked past yet.

    A row that fails the check is NOT deleted here - deciding that is an
    admin's job, and silently removing a gallery entry during a routine sweep
    is how a person stops being recognised with nothing to explain it. It is
    logged and reported by `added()`, so the review page can offer it.

    Returns the number of rows whose floor was written.
    """
    from sqlalchemy import update
    from app.db.models import FaceEmbedding
    from app.db.session import session_scope

    thr = settings.threshold_for(settings.recognizer_model)
    floor = max(thr, float(settings.augment_live_floor))
    M, owner, ids, tags, names, floors = _gallery_rows()
    live = [i for i, t in enumerate(tags) if t.startswith(TAG)]
    if not live:
        return 0

    idx = np.array(live, dtype=np.int64)
    P, near = corridor_probes(dim=int(M.shape[1]))
    if len(P):
        S = np.where(np.asarray(near)[None, :] == owner[idx][:, None], -2.0,
                     M[idx] @ np.asarray(P).T)
        imp = S.max(axis=1)
    else:
        imp = np.full(len(idx), -2.0, np.float32)

    with session_scope() as s:
        for k, i in enumerate(live):
            s.execute(update(FaceEmbedding)
                      .where(FaceEmbedding.id == ids[i])
                      .values(threshold=floor))
            if _floor(float(imp[k]), thr) > floor:
                log.warning(
                    "gallery: live row %d (%s) is reached by a real corridor "
                    "face at %.3f, at or above its %.3f floor - a lookalike. "
                    "Review it at /gallery/review; it is not removed here.",
                    ids[i], names.get(int(owner[i]), owner[i]), float(imp[k]), floor)
    log.info("gallery: %d live row(s) set to floor %.3f (%d corridor probes)",
             len(live), floor, len(P))
    return len(live)


def lookalikes() -> dict[int, float]:
    """Stored live rows a real corridor face already reaches above their floor.

    Same measurement `recalibrate` logs, returned so the review page can show
    which additions have since been contradicted by traffic. Keyed by
    `face_embedding.id`, valued by the similarity that reaches it.
    """
    thr = settings.threshold_for(settings.recognizer_model)
    floor = max(thr, float(settings.augment_live_floor))
    M, owner, ids, tags, _names, _floors = _gallery_rows()
    live = [i for i, t in enumerate(tags) if t.startswith(TAG)]
    if not live:
        return {}
    idx = np.array(live, dtype=np.int64)
    P, near = corridor_probes(dim=int(M.shape[1]))
    if not len(P):
        return {}
    S = np.where(np.asarray(near)[None, :] == owner[idx][:, None], -2.0,
                 M[idx] @ np.asarray(P).T)
    imp = S.max(axis=1)
    return {ids[i]: float(imp[k]) for k, i in enumerate(live)
            if _floor(float(imp[k]), thr) > floor}


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
    bad = lookalikes()
    return [{"id": r[0], "employee_id": r[1], "key": r[2][len(TAG):],
             "margin": r[3] or 0.0, "name": r[4], "threshold": r[5] or 0.0,
             "dead": (r[5] or 0.0) > settings.augment_max_threshold,
             # Set once real traffic has contradicted this crop: some other
             # face now reaches it above its floor. Shown so it can be removed.
             "reached": bad.get(r[0], 0.0)}
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
