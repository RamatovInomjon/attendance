"""Grouping the passes of people who were never enrolled.

THE QUESTION THIS ANSWERS
-------------------------
About half the passes through this corridor belong to somebody the face path
cannot name, and most of those people are not employees at all. Every one of
their passes is written as an independent "unknown", so the record can say how
many unrecognised passes happened and cannot say how many PEOPLE that was. One
visitor seen five times and five visitors seen once are the same row count.

This attaches each unnamed pass to a stable pseudo-identity - `P-000123` - so
the next pass of the same person joins the same one. That turns a pass count
into a visitor count, gives an unknown a history an operator can look at, and
makes "how long was this person in the building" a question with an answer.

WHAT IT IS NOT
--------------
It is not a way to name people. A pseudo-person becomes an employee ONLY when a
FACE template matches the enrolment gallery, and even then the naming relabels
the group rather than authoring attendance for passes that were observed
without an identity - the distinction app/services/corrections.py exists to
defend. Body similarity NEVER names anybody, at any threshold:

    of ~1100 unknown tracks a day, only ~45 are a recoverable employee.

At that base rate a body rule good enough to look useful (57% recall, 10.9% of
confirmed visitors mislabelled at the production threshold) writes 31 wrong
attendance rows for 27 right ones. Raising the threshold does not fix it -
at 0.90 it is 3 right and 2 wrong. Measured, three days, every threshold from
0.60 to 0.90: integration/docs/HISOBOT_YUZ_TANA.md, section 5.

THE TWO FEATURES, AND WHY IN THIS ORDER
---------------------------------------
    face  ->  clothing-independent, works ACROSS days.   PRIMARY
    body  ->  clothing-dependent, works within ONE day.  FALLBACK

Measured over three days and ~4200 unnamed passes (INTEGRATSIYA.md, 3.3):

    mode          pair precision   pair recall   pseudo-people   count error
    body alone       91-94%          3.5-7.6%      563/803/607    126-198%
    face alone       90-98%           23-31%      999/1240/1037      9-23%
    face + body      90-98%           29-32%       499/695/520       8-26%

Body alone cannot regroup anybody: two strangers in similar clothing out-score
one person seen twice. Its entire contribution is linking passes that have NO
usable face, and by doing that it halves the number of pseudo-people without
costing precision.

THE HONEST CEILING
------------------
Recall is about 30%, so one person's passes still land in roughly three groups.
The cause is not the model and not the tracker: only 23-28% of passes contain a
frame with an inter-pupil distance of 20 px or more, and face is the only key
that works. The distinct-visitor count is therefore accurate to about 15%.

That is fine for counting visitors and dwell time. It is nowhere near good
enough for attendance, which is why nothing here writes any.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
from sqlalchemy import select

from app.config import settings
from app.db.models import PseudoPerson, ReidPass

log = logging.getLogger(__name__)


def _unpack(blob: bytes | None, dim: int) -> np.ndarray:
    """Stored templates -> (K, D), or (0, D) when there are none.

    A blob that does not divide evenly by `dim` is a blob written under a
    different model. Returning it reshaped would be worse than returning
    nothing: the numbers would be finite, unit-ish and completely meaningless.
    """
    if not blob or dim <= 0:
        return np.zeros((0, max(dim, 1)), np.float32)
    v = np.frombuffer(blob, dtype=np.float32)
    if v.size % dim:
        return np.zeros((0, dim), np.float32)
    return v.reshape(-1, dim)


def _pack(rows: np.ndarray) -> bytes:
    return np.ascontiguousarray(rows, dtype=np.float32).tobytes()


def _push(blob: bytes | None, dim: int, v: np.ndarray, k: int) -> bytes:
    """Ring buffer: append `v`, keep the newest `k`."""
    cur = _unpack(blob, dim)
    v = np.asarray(v, np.float32).reshape(1, -1)
    return _pack(np.concatenate([cur, v])[-k:] if len(cur) else v)


def _best(q: np.ndarray | None, templates: list[np.ndarray]) -> np.ndarray:
    """Per candidate, the similarity of its CLOSEST template to the query.

    Closest, not mean: the templates of one person deliberately disagree -
    different distances, angles and, for the body, different moments of the
    walk. Averaging them is what turns a person's gallery into a generic one
    that resembles everybody slightly.
    """
    out = np.full(len(templates), -1.0, np.float32)
    if q is None:
        return out
    for i, T in enumerate(templates):
        if len(T):
            out[i] = float((T @ q).max())
    return out


class PseudoGallery:
    """Self-populating, open-set, online: it never sees the future.

    One instance per ReID worker thread; every call takes the caller's open
    session, so this owns no transaction and no lock of its own.

    IT MATCHES AGAINST AN IN-MEMORY INDEX, NOT AGAINST THE TABLE. The obvious
    implementation - select every recent pseudo-person and unpack its templates
    - runs once per unnamed pass, which is roughly once a second in this
    corridor. At the measured rate of 500-700 new pseudo-people a day, a
    two-week horizon is ~8000 rows carrying up to five 512-d face templates
    each; loading and unpacking them per pass would move tens of megabytes of
    BLOB through the ORM every time somebody walks past. So the face templates
    are held as ONE contiguous matrix with a parallel owner array - the same
    shape `app/core/gallery.py` uses, for the same reason - and matching is one
    matmul.

    THE DATABASE IS STILL THE SOURCE OF TRUTH. The index is rebuilt from it on
    first use and whenever the business date rolls over, so a restart loses
    nothing, and `place` writes to both.

    Memory is bounded by `pseudo_active_days` x the daily rate x
    `pseudo_templates_k`: at the measured numbers, roughly 40 MB of face
    templates for a fortnight. BODY templates are same-day only by design, so
    they are held in a small per-day dictionary and dropped at the rollover
    rather than being carried at all.
    """

    def __init__(self):
        self.assigned = 0
        self.created = 0
        self.by_face = 0
        self.by_body = 0
        self.named = 0
        # (N, D) unit-norm face templates and the pseudo-person each belongs to.
        self._face_M: np.ndarray | None = None
        self._face_owner = np.zeros((0,), np.int64)
        # pseudo id -> (K, D) body templates, for TODAY and this model only.
        self._body: dict[int, np.ndarray] = {}
        self._day: date | None = None
        self._body_model = ""

    # -- the index --------------------------------------------------------
    def _load(self, s, today: date, body_model: str) -> None:
        """(Re)build the index from the table. Once at start, once per day.

        The horizon is deliberate and is not only about cost: without one, a
        visitor from six months ago stays a live candidate forever, so a rare
        face collision becomes a permanent wrong grouping rather than a
        transient one.
        """
        # Keyed on the day AND the model. Not on `self._face_M is not None`:
        # a day whose first passes carry no face at all would then re-run this
        # query on every pass, which is the cost the index exists to remove.
        if self._day == today and self._body_model == body_model:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=max(1, settings.pseudo_active_days))
        rows = s.execute(
            select(PseudoPerson.id, PseudoPerson.face_templates,
                   PseudoPerson.face_dim, PseudoPerson.body_templates,
                   PseudoPerson.body_dim, PseudoPerson.body_model,
                   PseudoPerson.body_date)
            .where(PseudoPerson.last_seen >= cutoff)
        ).all()

        blocks, owners = [], []
        body: dict[int, np.ndarray] = {}
        for pid, fblob, fdim, bblob, bdim, bmodel, bdate in rows:
            T = _unpack(fblob, fdim or 0)
            if len(T):
                blocks.append(T)
                owners.append(np.full(len(T), pid, np.int64))
            # Only today's body templates are ever comparable - see the module
            # note on clothing - so yesterday's are not even loaded.
            if bmodel == body_model and (bdate == today
                                         or not settings.pseudo_body_same_day_only):
                B = _unpack(bblob, bdim or 0)
                if len(B):
                    body[pid] = B
        self._face_M = (np.ascontiguousarray(np.concatenate(blocks))
                        if blocks else None)
        self._face_owner = (np.concatenate(owners) if owners
                            else np.zeros((0,), np.int64))
        self._body = body
        self._day = today
        self._body_model = body_model
        log.info("[pseudo] index: %d face template(s) over %d people, "
                 "%d with a body template for %s",
                 0 if self._face_M is None else len(self._face_M),
                 int(len(np.unique(self._face_owner))), len(body), today)

    def _remember(self, pid: int, face, body) -> None:
        """Fold one pass's templates into the index, mirroring the row write."""
        if face is not None:
            v = np.asarray(face, np.float32).reshape(1, -1)
            if self._face_M is None or self._face_M.shape[1] != v.shape[1]:
                self._face_M, self._face_owner = v.copy(), np.array([pid], np.int64)
            else:
                self._face_M = np.concatenate([self._face_M, v])
                self._face_owner = np.concatenate(
                    [self._face_owner, np.array([pid], np.int64)])
        if body is not None:
            v = np.asarray(body, np.float32).reshape(1, -1)
            cur = self._body.get(pid)
            k = max(1, settings.pseudo_templates_k)
            self._body[pid] = (v if cur is None or cur.shape[1] != v.shape[1]
                               else np.concatenate([cur, v])[-k:])

    # -- matching ---------------------------------------------------------
    def _search(self, face, body) -> tuple[int, float, str]:
        """(pseudo id, score, which feature decided), or (-1, 0.0, "").

        FACE FIRST, and body is still tried when face FAILS rather than only
        when face is absent. Face quality varies frame to frame; one pass whose
        best face happens to be poor should not break a chain the body can
        still carry within the same day.
        """
        if face is not None and self._face_M is not None \
                and self._face_M.shape[1] == len(face):
            sims = self._face_M @ np.asarray(face, np.float32)
            i = int(sims.argmax())
            if float(sims[i]) >= settings.pseudo_face_threshold:
                return int(self._face_owner[i]), float(sims[i]), "face"

        if body is None or not self._body:
            return -1, 0.0, ""
        ids = list(self._body)
        sims = _best(np.asarray(body, np.float32),
                     [self._body[p] for p in ids])
        i = int(sims.argmax())
        if float(sims[i]) < settings.pseudo_body_threshold:
            return -1, 0.0, ""
        # The runner-up rule, for the reason it exists everywhere else here:
        # clothing similarity is common, so a top score that two candidates
        # both nearly reach is evidence of a coat, not of a person.
        second = (float(np.partition(sims, -2)[-2]) if len(sims) > 1 else -1.0)
        if (float(sims[i]) - second) < settings.pseudo_body_margin:
            return -1, 0.0, ""
        return ids[i], float(sims[i]), "body"

    # -- the entry point --------------------------------------------------
    def place(self, s, row: ReidPass, *, face=None, body=None,
              face_ipd: float = 0.0, body_model: str = "",
              registry_match=None, create: bool = True) -> PseudoPerson | None:
        """Put one finished pass into the gallery.

        `registry_match` is `(employee_id, score, name)` when the face path
        already named this pass, or None. It is passed in rather than computed
        here so that the ONE gallery lookup the worker already performs is what
        decides, and this module never becomes a second, differently calibrated
        way of naming somebody.

        `create=False` means "link, or do nothing". That is what a NAMED pass
        gets. A recognised employee arriving here is not a new visitor and must
        not mint a pseudo-identity - the count would stop meaning anything -
        but their face may well match a pseudo-person built from their own
        earlier passes, the ones the face path missed. That is the only path by
        which a group of faceless unknowns ever gets a name, because a pass
        with a face good enough to match the registry is named upstream and
        never arrives here as an unknown.

        Returns the pseudo-person the pass joined, or None when it was not
        placed - which happens when the pass has neither a usable face nor a
        body feature to offer, or when `create` is False and nothing matched.
        Neither is a failure.
        """
        # A face below the quality floor is DISCARDED for matching, not merely
        # down-weighted. Measured on this corridor, a face under ~20 px of
        # inter-pupil distance is a source of wrong links: it will happily
        # reach 0.35 against a stranger. The pass falls through to body.
        if face is not None and face_ipd < settings.pseudo_face_ipd_min:
            face = None
        if face is None and body is None:
            return None

        today = row.business_date
        self._load(s, today, body_model)
        pid, score, how = self._search(face, body)

        if pid < 0 and not create:
            return None
        if pid < 0:
            p = PseudoPerson(
                code="",                       # assigned from the id below
                first_seen=row.first_seen, last_seen=row.last_seen,
                n_passes=0, face_dim=0, body_dim=0, body_model="",
            )
            s.add(p)
            s.flush()                          # need the id to mint the code
            p.code = f"P-{p.id:06d}"
            how = "new"
            self.created += 1
        else:
            # One row, by primary key - not the thousands the scan would have
            # loaded to find it.
            p = s.get(PseudoPerson, pid)
            if p is None:
                # The index outlived its row (a manual delete, a restored
                # database). Rebuild rather than write against a ghost.
                self._day = None
                return None
            self.assigned += 1
            if how == "face":
                self.by_face += 1
            else:
                self.by_body += 1

        # -- fold this pass's templates in --------------------------------
        k = max(1, settings.pseudo_templates_k)
        if face is not None:
            # A dimension change means the recognizer was swapped, and the
            # stored templates are in a space this vector knows nothing about.
            # Start over rather than concatenate two incomparable geometries.
            fresh = p.face_dim != int(len(face))
            p.face_templates = (_pack(np.asarray(face, np.float32).reshape(1, -1))
                                if fresh else _push(p.face_templates, p.face_dim,
                                                    face, k))
            p.face_dim = int(len(face))
        if body is not None:
            # A different model, or a different day, invalidates what is there:
            # keeping it would leave a stale template able to match tomorrow.
            stale = (p.body_model != body_model
                     or p.body_dim != int(len(body))
                     or (settings.pseudo_body_same_day_only
                         and p.body_date != today))
            p.body_templates = (_pack(np.asarray(body, np.float32).reshape(1, -1))
                                if stale or not p.body_dim
                                else _push(p.body_templates, p.body_dim, body, k))
            p.body_dim = int(len(body))
            p.body_model = body_model
            p.body_date = today

        self._remember(p.id, face, body)

        p.n_passes = (p.n_passes or 0) + 1
        p.last_seen = max(p.last_seen, row.last_seen) if p.last_seen else row.last_seen
        if p.first_seen is None or row.first_seen < p.first_seen:
            p.first_seen = row.first_seen

        row.pseudo_person_id = p.id
        row.pseudo_score = float(score)
        row.pseudo_by = how

        # -- naming, by FACE and the registry only ------------------------
        if registry_match is not None and p.employee_id is None:
            emp, sc, who = registry_match
            p.employee_id, p.named_score = int(emp), float(sc)
            p.named_at = datetime.now(timezone.utc)
            self.named += 1
            # Relabel the history so an operator reviewing this group sees a
            # name rather than a code. This writes NO attendance: those passes
            # were observed without an identity, and inventing transitions for
            # them after the fact is the fabrication corrections.py refuses.
            back = s.execute(
                select(ReidPass).where(ReidPass.pseudo_person_id == p.id,
                                       ReidPass.employee_id.is_(None))
            ).scalars().all()
            for r in back:
                r.employee_id, r.name = int(emp), who
            log.info("[pseudo] %s is employee %d (face %.3f); %d earlier pass(es) "
                     "relabelled - no attendance written", p.code, emp, sc, len(back))

        return p

    def stats(self) -> dict:
        return {"assigned": self.assigned, "created": self.created,
                "by_face": self.by_face, "by_body": self.by_body,
                "named": self.named}
