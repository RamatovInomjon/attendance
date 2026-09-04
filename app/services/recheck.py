"""Re-decide recognitions that were made under an older gallery, and fix them.

WHY THIS EXISTS
---------------
A recognition is a decision taken once, against whatever the gallery and the
threshold were at that instant. Change either and the record does not change
with it: a pass accepted at 0.251 through an unfloored corridor crop stays in
the attendance table forever, naming the wrong person, long after the rule that
admitted it has been replaced.

That is not hypothetical here. On 03-09 a guard in a black uniform was recorded
as Xamdamov Rustam checking out at 13:00, on the strength of one corridor crop
at 0.251 and nothing else - no enrolment photograph of Xamdamov came near him.
The 0.35 floor deployed on 04-09 rejects that face outright. It could not
reach back and un-say it.

So this replays the stored passes against the CURRENT gallery, and voids the
ones the system would no longer make. Every void goes through
`corrections.void_event`, so the day is rebuilt by the same replay an admin's
button uses - which means a promoted check-out, a recomputed `worked_seconds`,
and the poisoning crop removed, all for free.

WHAT IT WILL AND WILL NOT DO
----------------------------
It only ever REMOVES a name. A pass the current gallery would accept and the
old one did not is not invented here: that recognition never happened, nobody
observed it, and writing attendance for a walk the system did not see at the
time is fabrication however good the arithmetic.

It only judges a pass it can SEE. The evidence is the `data/debug` capture, and
if `debug_capture` was off or retention has swept it, the pass is left alone and
reported as unjudged. Voiding on the basis of a decision we cannot re-derive
would be worse than the error it is trying to fix.

One approximation, stated plainly: a capture is the best-scoring frame of its
pass, not the whole pass. That makes this a sound test of REFUSAL - if the best
frame clears nothing, no weaker frame of the same pass does - and not a re-run
of the consensus vote. It is why this can only remove.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)


@dataclass
class Verdict:
    stem: str
    event_id: int | None
    employee_id: int
    name: str
    camera: str
    business_date: date | None
    transition: str
    score_then: float          # what the pass scored when it was accepted
    score_now: float           # best per-person score under today's rules
    enrolment_best: float      # what the studio photographs alone reach
    still_named: bool
    already_voided: bool = False
    face: str = ""

    @property
    def refused(self) -> bool:
        return not self.still_named and not self.already_voided


@dataclass
class Report:
    verdicts: list = field(default_factory=list)
    captures: int = 0
    unjudged: int = 0          # a stored event with no capture left to judge by

    @property
    def refused(self):
        return [v for v in self.verdicts if v.refused]


def _decide(row_scores, owner, people, slot, floors, thr, margin, use_floors):
    """Per-person max then the two acceptance rules, mirroring Gallery."""
    s = row_scores - floors + thr if use_floors else row_scores
    out = np.full(len(people), -4.0, np.float32)
    np.maximum.at(out, slot, s)
    if len(people) == 1:
        return (int(people[0]) if out[0] >= thr else None), float(out[0])
    idx = np.argpartition(out, -2)[-2:]
    if out[idx[0]] > out[idx[1]]:
        idx = idx[::-1]
    second, top = int(idx[0]), int(idx[1])
    if out[top] >= thr and (out[top] - out[second]) >= margin:
        return int(people[top]), float(out[top])
    return None, float(out[top])


def _event_index(since: date | None):
    """Stored events keyed the way a debug capture is named.

    `<YYYYMMDD_HHMMSS>_<score:.3f>_<camera>` in LOCAL time. Built by walking
    the events once rather than globbing the directory per event, which turned
    a two-second pass into a per-row filesystem search.
    """
    from sqlalchemy import select
    from app.db.models import Camera, RecognitionEvent
    from app.db.session import session_scope

    with session_scope() as s:
        q = select(RecognitionEvent.id, RecognitionEvent.ts, RecognitionEvent.score,
                   RecognitionEvent.employee_id, RecognitionEvent.business_date,
                   RecognitionEvent.transition, RecognitionEvent.voided_at,
                   Camera.name).outerjoin(Camera, Camera.id == RecognitionEvent.camera_id)
        if since is not None:
            q = q.where(RecognitionEvent.business_date >= since)
        rows = s.execute(q).all()
    out = {}
    for r in rows:
        local = r[1].astimezone(settings.tz)
        out[f"{local:%Y%m%d_%H%M%S}_{float(r[2] or 0):.3f}_{r[7] or ''}"] = r
    return out


def run(since: date | None = None, limit: int = 0) -> Report:
    """Re-decide every stored pass we still have the evidence for."""
    import cv2
    from app.core.geometry import to_normalized_chw
    from app.core.recognizer import FaceRecognizer
    from app.services.augment import TAG, _gallery_rows

    rep = Report()
    thr = settings.threshold_for(settings.recognizer_model)
    margin = settings.second_best_margin
    M, owner, _ids, tags, names, floors = _gallery_rows()
    if not len(M):
        # No gallery, no verdicts. Every pass would "fail" against nothing, and
        # voiding the entire history because the embeddings had not loaded is
        # the worst thing this could do.
        log.warning("recheck: the gallery is empty; nothing re-decided")
        return rep
    live = np.array([t.startswith(TAG) for t in tags])
    people = np.unique(owner)
    slot = np.searchsorted(people, owner).astype(np.int64)
    index = _event_index(since)

    files = sorted(glob.glob(str(settings.debug_dir / "**" / "*.json"), recursive=True))
    caps, meta = [], []
    for jf in files:
        aligned = jf[:-5] + "_aligned.jpg"
        if not os.path.exists(aligned):
            continue
        try:
            m = json.loads(open(jf).read())
        except Exception:
            continue
        if m.get("employee_id") is None:
            continue
        stem = Path(jf).stem
        key = stem.rsplit("_", 1)[0]          # drop the trailing sequence number
        ev = index.get(key)
        if ev is None:
            continue                          # no stored event; nothing to fix
        if since is not None and ev[4] and ev[4] < since:
            continue
        im = cv2.imread(aligned)
        if im is None:
            continue
        if im.shape[:2] != (112, 112):
            im = cv2.resize(im, (112, 112), interpolation=cv2.INTER_AREA)
        caps.append(to_normalized_chw(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        meta.append((stem, m, ev, jf[:-5] + "_face.jpg"))
    if limit:
        caps, meta = caps[-limit:], meta[-limit:]
    rep.captures = len(caps)
    # Counted BEFORE the early return. A run that finds no captures at all -
    # `debug_capture` switched off, or retention having swept the window - must
    # still say how many stored passes it could not judge. Reporting a clean
    # "nothing to correct" in that case would be the most misleading output
    # this could produce.
    judged = {m[2][0] for m in meta}
    rep.unjudged = sum(1 for r in index.values()
                       if r[0] not in judged and r[6] is None)
    if not caps:
        return rep

    rec = FaceRecognizer(settings.model_path(settings.recognizer_model),
                         batch_size=settings.embed_batch)
    Q = rec.embed(np.stack(caps))
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
    S = Q @ M.T

    for i, (stem, m, ev, face) in enumerate(meta):
        who, now = _decide(S[i], owner, people, slot, floors, thr, margin, True)
        eo = np.where(~live, S[i], -2.0)
        rep.verdicts.append(Verdict(
            stem=stem, event_id=int(ev[0]), employee_id=int(ev[3]),
            name=names.get(int(ev[3]), str(ev[3])), camera=ev[7] or "",
            business_date=ev[4], transition=ev[5] or "",
            score_then=float(m.get("score", 0.0)), score_now=now,
            enrolment_best=float(eo.max()) if (~live).any() else -1.0,
            # Still named means named AS THE SAME PERSON. A pass that now names
            # somebody else is not a confirmation; it is a different error, and
            # it is refused rather than quietly re-assigned - moving a name from
            # one employee to another is not something a batch job should do.
            still_named=(who is not None and who == int(ev[3])),
            already_voided=ev[6] is not None, face=face))
    return rep


def apply(report: Report, *, by: str, reason: str = "") -> dict:
    """Void every refused pass, rebuilding each affected day exactly once."""
    from app.services import corrections
    done, failed, days = 0, [], set()
    for v in report.refused:
        out = corrections.void_event(
            v.event_id, by=by,
            reason=reason or f"recheck: {v.score_then:.3f} -> {v.score_now:.3f}")
        if out.get("ok"):
            done += 1
            days.add((v.employee_id, str(v.business_date)))
        else:
            failed.append((v.event_id, out.get("error", "")))
    log.info("recheck: voided %d pass(es) across %d person-days", done, len(days))
    return {"voided": done, "days": len(days), "failed": failed}
