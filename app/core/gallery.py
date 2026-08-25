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
    best_score: float = 0.0
    best_snapshot: np.ndarray | None = None      # highest-SCORING frame (diagnosis)
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
        if m.employee_id is not None and m.score > self.best_score:
            self.best_score = m.score
            if snapshot is not None:
                self.best_snapshot = snapshot
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

    @property
    def decided(self) -> bool:
        return self.committed is not None
