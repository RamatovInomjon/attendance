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
  * the code then re-measures the gallery's worst impostor pair with the
    selection included, and REFUSES if it reaches the recognition threshold -
    naming the two identities responsible. A human can be fooled by a bad crop
    of a lookalike; this catches what the eye misses.

This module is the single implementation of both. `scripts/augment_gallery.py`
and the web review page call in here, so the CLI and the UI cannot enforce
different rules.
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
    vec: np.ndarray | None = field(default=None, repr=False)
    rejected: str = ""             # why it is not offered, if it is not


@dataclass
class ImpostorCheck:
    before: float
    after: float
    threshold: float
    pair: tuple[str, str] | None = None

    @property
    def safe(self) -> bool:
        return self.after < self.threshold


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
    out.sort(key=lambda c: (c.name, -c.margin))
    return out


def check_impostors(chosen: list[Candidate]) -> ImpostorCheck:
    """Would adding these make two different people match each other?

    This is the check the whole feature turns on. Augmentation is meant to
    raise genuine scores WITHOUT raising impostor scores; if the worst
    cross-identity pair reaches the recognition threshold, the additions are
    wrong however good they looked.
    """
    from sqlalchemy import select
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope

    thr = settings.threshold_for(settings.recognizer_model)
    with session_scope() as s:
        rows = s.execute(select(FaceEmbedding.employee_id,
                                FaceEmbedding.vector)).all()
        names = dict(s.execute(select(Employee.id, Employee.full_name)).all())
    if not rows:
        return ImpostorCheck(-1.0, -1.0, thr)

    M = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    owner = np.array([r[0] for r in rows])
    labels = [f"enrolment · {names.get(int(e), e)}" for e in owner]

    def worst(mat, own, lab):
        sim = mat @ mat.T
        np.fill_diagonal(sim, -1)
        masked = np.where(own[:, None] != own[None, :], sim, -1)
        i, j = np.unravel_index(int(masked.argmax()), masked.shape)
        return float(masked[i, j]), (lab[i], lab[j])

    before, _ = worst(M, owner, labels)
    if not chosen:
        return ImpostorCheck(before, before, thr)
    M2 = np.vstack([M, np.stack([c.vec for c in chosen])])
    o2 = np.concatenate([owner, np.array([c.employee_id for c in chosen])])
    lab2 = labels + [f"NEW · {c.name} ({c.when}, {c.camera})" for c in chosen]
    after, pair = worst(M2, o2, lab2)
    return ImpostorCheck(before, after, thr, pair)


def add(chosen: list[Candidate]) -> int:
    """Insert the chosen crops, tagged so they can be found and removed."""
    from app.db.models import FaceEmbedding
    from app.db.session import session_scope
    if not chosen:
        return 0
    with session_scope() as s:
        for c in chosen:
            s.add(FaceEmbedding(
                employee_id=c.employee_id, source_file=f"{TAG}{c.key}",
                vector=c.vec.astype(np.float32).tobytes(),
                dim=int(c.vec.shape[0]),
                model_name=settings.recognizer_model, quality=float(c.margin)))
    log.info("gallery: added %d live embedding(s)", len(chosen))
    return len(chosen)


def added() -> list[dict]:
    """Everything previously added from the corridor, newest first."""
    from sqlalchemy import select
    from app.db.models import Employee, FaceEmbedding
    from app.db.session import session_scope
    with session_scope() as s:
        rows = s.execute(
            select(FaceEmbedding.id, FaceEmbedding.employee_id,
                   FaceEmbedding.source_file, FaceEmbedding.quality,
                   Employee.full_name)
            .join(Employee, Employee.id == FaceEmbedding.employee_id)
            .where(FaceEmbedding.source_file.like(f"{TAG}%"))
            .order_by(FaceEmbedding.id.desc())).all()
    return [{"id": r[0], "employee_id": r[1], "key": r[2][len(TAG):],
             "margin": r[3] or 0.0, "name": r[4]} for r in rows]


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
    log.info("gallery: removed %d live embedding(s)", r.rowcount)
    return int(r.rowcount)
