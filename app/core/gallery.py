"""Gallery matching and per-track identity voting.

Matching is one matrix product against every enrolment embedding, then a
max-reduce per person.  The previous code walked a Python list and recomputed
`emb / norm(emb)` for every candidate on every frame — a constant, recomputed
thousands of times a second.  At 54 people x 5 images this is ~270 rows: the
matmul is microseconds.

Two acceptance rules, both required:

* **threshold** — best similarity must clear `RECOGNITION_THRESHOLD`
* **margin**    — best must beat the runner-up *person* by `SECOND_BEST_MARGIN`

The margin rule is what catches lookalikes.  On this gallery the closest
impostor pair (007_Azizbek / 036_Anvar) peaks at 0.397, so it rarely fires — but
it is the rule that degrades gracefully as the gallery grows.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Match:
    employee_id: int | None
    score: float
    margin: float
    runner_up: int | None = None


class Gallery:
    """Immutable snapshot of the enrolment set, rebuilt on change."""

    def __init__(self, vectors: np.ndarray, employee_ids: np.ndarray, names: dict[int, str]):
        if len(vectors) == 0:
            self.M = np.zeros((0, 512), dtype=np.float32)
            self.owner = np.zeros((0,), dtype=np.int64)
        else:
            M = np.asarray(vectors, dtype=np.float32)
            M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12   # normalize once
            self.M = np.ascontiguousarray(M)
            self.owner = np.asarray(employee_ids, dtype=np.int64)
        self.names = names
        self._people = np.unique(self.owner) if len(self.owner) else np.zeros((0,), np.int64)

    def __len__(self):
        return len(self.M)

    @property
    def n_people(self) -> int:
        return len(self._people)

    def name(self, employee_id: int | None) -> str:
        return self.names.get(employee_id, "Unknown") if employee_id is not None else "Unknown"

    def match(self, embedding: np.ndarray, threshold: float, margin: float) -> Match:
        """Best person for one query embedding."""
        if len(self.M) == 0:
            return Match(None, 0.0, 0.0)

        q = np.asarray(embedding, dtype=np.float32).ravel()
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = self.M @ q                              # (N,)

        # Max similarity per person, not per image.
        best_per_person: dict[int, float] = {}
        for owner, s in zip(self.owner, sims):
            o = int(owner)
            if s > best_per_person.get(o, -2.0):
                best_per_person[o] = float(s)

        ranked = sorted(best_per_person.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_s = ranked[0]
        second_id, second_s = (ranked[1] if len(ranked) > 1 else (None, -1.0))
        gap = top_s - second_s if second_id is not None else top_s

        if top_s >= threshold and gap >= margin:
            return Match(top_id, top_s, gap, second_id)
        return Match(None, top_s, gap, top_id)      # runner_up carries the near miss


@dataclass
class TrackVote:
    """K-of-N agreement over a track's best frames.

    A single frame never commits an identity.  The old pipeline recognized the
    first frame a track appeared in and locked that answer for an hour — a
    motion-blurred profile at the moment of entry decided who the person was.
    """
    window: int
    required: int
    votes: deque = field(default_factory=deque)
    # Best score and frame PER IDENTITY.  Keyed by employee id, so the evidence
    # for the person we commit can never be another person's frame - see the
    # note on best_snapshot below.
    per_identity: dict = field(default_factory=dict)   # emp_id -> [score, frame, margin]
    best_quality: float = 0.0
    quality_snapshot: np.ndarray | None = None   # clearest frame (what a human sees)
    committed: int | None = None

    def __post_init__(self):
        self.votes = deque(maxlen=self.window)

    def add(self, m: Match, snapshot: np.ndarray | None = None,
            quality: float = 0.0) -> int | None:
        """Record one observation; returns an employee id once agreement is met.

        Two snapshots are kept, deliberately. The highest-SCORING frame explains
        why the match happened and belongs in diagnostics. The highest-QUALITY
        frame is what a person should be shown, because on a false positive the
        top-scoring frame is precisely the one that spuriously resembled the
        wrong person - a forehead crop that happened to score 0.16. Showing that
        as the evidence makes a wrong answer look inexplicable.
        """
        self.votes.append(m.employee_id)
        if m.employee_id is not None:
            cur = self.per_identity.get(m.employee_id)
            if cur is None or m.score > cur[0]:
                self.per_identity[m.employee_id] = [
                    m.score, snapshot if snapshot is not None
                    else (cur[1] if cur is not None else None),
                    m.margin]
        if snapshot is not None and quality > self.best_quality:
            self.best_quality = quality
            self.quality_snapshot = snapshot

        if self.committed is not None:
            return self.committed

        counts = Counter(v for v in self.votes if v is not None)
        if counts:
            emp, n = counts.most_common(1)[0]
            if n >= self.required:
                self.committed = emp
                return emp
        return None

    def best_for(self, employee_id: int | None) -> tuple[float, np.ndarray | None]:
        """Best score and frame observed FOR ONE identity."""
        v = self.per_identity.get(employee_id)
        return (v[0], v[1]) if v is not None else (0.0, None)

    def _global_best(self) -> tuple[float, np.ndarray | None]:
        if not self.per_identity:
            return 0.0, None
        return tuple(max(self.per_identity.values(), key=lambda v: v[0]))

    @property
    def best_score(self) -> float:
        """The score behind the committed identity, not the track's maximum.

        These used to be a single running maximum over every frame, regardless
        of WHICH person each frame matched.  When one track saw two people -
        two colleagues walking together, or a ByteTrack id switch - the vote
        would commit person A by majority while the maximum belonged to a frame
        of person B.  The event then carried A's name with B's score and B's
        face, which is exactly how a correct-looking recognition ends up
        displaying a stranger.  Observed live: track 601 committed one employee
        while reporting 0.277 and an aligned crop that were another employee's.
        """
        if self.committed is not None:
            return self.best_for(self.committed)[0]
        return self._global_best()[0]

    @property
    def best_snapshot(self) -> np.ndarray | None:
        """The highest-scoring frame FOR the committed identity.

        Before a commit there is no identity to scope to, so the running best
        is the only thing available; after one, the answer is definitionally
        the committed person's own best frame.
        """
        if self.committed is not None:
            snap = self.best_for(self.committed)[1]
            if snap is not None:
                return snap
        return self._global_best()[1]

    @property
    def best_margin(self) -> float:
        """Gap to the runner-up on the frame that decided the identity.

        The single most informative number for judging a false accept: a score
        of 0.216 means one thing when the next candidate was 0.05 behind and
        quite another when it was 0.002 behind.  It was computed at match time
        and then thrown away - every stored event read margin 0.0 - so there
        was no way to tell the two apart after the fact.
        """
        v = self.per_identity.get(self.committed) if self.committed is not None else None
        if v is None:
            g = self._global_best()
            return float(g[2]) if len(g) > 2 else 0.0
        return float(v[2])

    @property
    def decided(self) -> bool:
        return self.committed is not None
