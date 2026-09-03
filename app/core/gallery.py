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
    """Immutable snapshot of the enrolment set, rebuilt on change.

    `thresholds` is a PER-IMAGE acceptance floor, and is the mechanism behind
    corridor augmentation (app/services/augment.py).  An enrolment photograph
    is judged against the global threshold and passes 0.0 here.  A crop lifted
    from the corridor is judged against its own, higher floor: the similarity
    it was measured to reach against the nearest *other* person, plus a safety
    margin.  Below that floor the row cannot name anybody, so a live crop is
    incapable of manufacturing the false accept a single shared threshold would
    have allowed - while still firing at the lowest similarity that is
    demonstrably safe for that particular face.
    """

    def __init__(self, vectors: np.ndarray, employee_ids: np.ndarray,
                 names: dict[int, str], thresholds: np.ndarray | None = None):
        if len(vectors) == 0:
            self.M = np.zeros((0, 512), dtype=np.float32)
            self.owner = np.zeros((0,), dtype=np.int64)
        else:
            # copy=True, not np.asarray: asarray hands back the CALLER'S array
            # when it is already float32, and the `/=` below then normalises
            # their data under them. Enrolment and the benchmarks both keep the
            # array they passed in.
            M = np.array(vectors, dtype=np.float32, copy=True)
            M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12   # normalize once
            self.M = np.ascontiguousarray(M)
            self.owner = np.asarray(employee_ids, dtype=np.int64)
        self.names = names
        self._people = np.unique(self.owner) if len(self.owner) else np.zeros((0,), np.int64)
        # Dense 0..P-1 index per row, so the per-person max is a scatter-reduce
        # instead of a Python loop over every enrolment image. See `match`.
        if len(self.owner):
            self._slot = np.searchsorted(self._people, self.owner).astype(np.int64)
        else:
            self._slot = np.zeros((0,), np.int64)
        self._MT = np.ascontiguousarray(self.M.T)
        # None when every row is judged against the global threshold, which is
        # every gallery that has never been augmented. That keeps the matching
        # hot path byte-for-byte what it was.
        self._floor = None
        if thresholds is not None and len(self.M):
            f = np.nan_to_num(np.asarray(thresholds, dtype=np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
            if f.shape != (len(self.M),):
                raise ValueError(
                    f"thresholds has {f.shape} entries for {len(self.M)} "
                    f"embeddings - they are positional and must line up")
            if np.any(f > 0):
                self._floor = f
        self._pen: np.ndarray | None = None
        self._pen_for: float | None = None

    def __len__(self):
        return len(self.M)

    @property
    def n_people(self) -> int:
        return len(self._people)

    def name(self, employee_id: int | None) -> str:
        return self.names.get(employee_id, "Unknown") if employee_id is not None else "Unknown"

    def _decide(self, per_person: np.ndarray, threshold: float, margin: float) -> Match:
        """Apply the two acceptance rules to one person-scored row."""
        if len(per_person) == 1:
            top = 0
            top_s = float(per_person[0])
            return (Match(int(self._people[0]), top_s, top_s, None)
                    if top_s >= threshold else Match(None, top_s, top_s,
                                                     int(self._people[0])))
        # argpartition, not a full sort: only the top two matter, and this is
        # the one part of matching that grows with headcount.
        idx = np.argpartition(per_person, -2)[-2:]
        if per_person[idx[0]] > per_person[idx[1]]:
            idx = idx[::-1]
        second, top = int(idx[0]), int(idx[1])
        top_s, second_s = float(per_person[top]), float(per_person[second])
        gap = top_s - second_s
        if top_s >= threshold and gap >= margin:
            return Match(int(self._people[top]), top_s, gap, int(self._people[second]))
        return Match(None, top_s, gap, int(self._people[top]))   # near miss

    def _penalty(self, threshold: float) -> np.ndarray:
        """How much each row's own floor exceeds the global threshold.

        Clamped at zero, so a stored floor can only ever make a row HARDER to
        match. A row whose floor somehow landed below the global threshold
        would otherwise become a private back door into the gallery, which is
        the exact failure this machinery exists to prevent.
        """
        if self._pen_for != threshold:
            self._pen = np.maximum(self._floor - threshold, 0.0).astype(np.float32)
            self._pen_for = threshold
        return self._pen

    def _per_person(self, sims: np.ndarray, threshold: float) -> np.ndarray:
        """(..., N images) similarities -> (..., P people) best-image-per-person.

        This replaced a Python loop over every enrolment row. The matmul above
        it is microseconds; the loop was ~50x that at 268 images, and it grew
        with the gallery - the one part of matching that does.

        Rows carrying their own floor are shifted DOWN by the amount their
        floor exceeds the global threshold, so that everything downstream keeps
        comparing against one number. "row i clears its own floor" and "the
        shifted score clears the global threshold" are the same statement, and
        expressing it as a shift means the per-person max and the runner-up
        margin both rank rows on one calibrated scale instead of comparing a
        strict row's score with a lenient row's on equal terms.
        """
        if self._floor is not None:
            sims = sims - self._penalty(threshold)
        # -4.0, not -2.0: a shifted score bottoms out at -1 - max(penalty), and
        # the fill has to stay below anything a real row can produce.
        out = np.full(sims.shape[:-1] + (len(self._people),), -4.0, dtype=np.float32)
        np.maximum.at(out, (Ellipsis, self._slot), sims)
        return out

    def match(self, embedding: np.ndarray, threshold: float, margin: float) -> Match:
        """Best person for one query embedding."""
        if len(self.M) == 0:
            return Match(None, 0.0, 0.0)
        q = np.asarray(embedding, dtype=np.float32).ravel()
        q = q / (np.linalg.norm(q) + 1e-12)
        return self._decide(self._per_person(self.M @ q, threshold), threshold, margin)

    def match_batch(self, embeddings: np.ndarray, threshold: float,
                    margin: float) -> list[Match]:
        """Every face in one frame, in one GEMM.

        Detection, alignment and embedding are all batched per frame; matching
        was the only stage still looping in Python, once per face.
        """
        E = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        if len(self.M) == 0:
            return [Match(None, 0.0, 0.0) for _ in range(len(E))]
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        per = self._per_person(E @ self._MT, threshold)           # (K, P)
        return [self._decide(per[i], threshold, margin) for i in range(len(E))]


@dataclass
class TrackVote:
    """Consensus over EVERY recognition of one track.

    A single frame never commits an identity.  The original pipeline recognized
    the first frame a track appeared in and locked that answer for an hour — a
    motion-blurred profile at the moment of entry decided who the person was.
    K-of-N agreement fixed that but was still "first to 3 of the last 5": an
    identity could be settled from the opening frames of a pass and never
    revisited, even though a pass yields dozens of frames and the opening ones
    are not reliably its best.  Live scores for one person spanned 0.196-0.473
    within a single day.

    So the decision now waits for the whole pass:

    * every frame votes, into an unbounded tally;
    * `provisional_id` is the current leader, for the live overlay only;
    * `finalize()` at track end applies the consensus rule and yields the one
      answer attendance is allowed to use.

    The rule is `vote_consensus` of the IDENTIFIED frames, and at least
    `vote_min_recognitions` of them.  See app/config.py for why misses are
    excluded from the denominator and why the floor exists.

    This is also the strongest available defence against a false accept.  An
    impostor has no true identity in the gallery, so their frames scatter over
    whoever is nearest by noise; a real person's frames converge.  Consensus
    measures precisely that, and it is a different axis from score - which is
    unusable alone here, two confirmed impostors having scored 0.218 and 0.180
    inside a genuine range reaching down to 0.177.
    """
    window: int
    required: int
    consensus: float = 0.65
    min_recognitions: int = 5
    votes: deque = field(default_factory=deque)        # recent window, for display
    tally: Counter = field(default_factory=Counter)    # UNBOUNDED, identified only
    identified: int = 0                                # frames that named somebody
    missed: int = 0                                    # frames that named nobody
    # Best score and frame PER IDENTITY.  Keyed by employee id, so the evidence
    # for the person we commit can never be another person's frame - see the
    # note on best_snapshot below.
    per_identity: dict = field(default_factory=dict)   # emp_id -> [score, face, margin, body]
    best_quality: float = 0.0
    quality_snapshot: np.ndarray | None = None   # clearest frame (what a human sees)
    committed: int | None = None
    finalized: bool = False

    def __post_init__(self):
        self.votes = deque(maxlen=self.window)

    def add(self, m: Match, snapshot: np.ndarray | None = None,
            quality: float = 0.0, person: np.ndarray | None = None) -> int | None:
        """Record one observation; returns an employee id once agreement is met.

        Two snapshots are kept, deliberately. The highest-SCORING frame explains
        why the match happened and belongs in diagnostics. The highest-QUALITY
        frame is what a person should be shown, because on a false positive the
        top-scoring frame is precisely the one that spuriously resembled the
        wrong person - a forehead crop that happened to score 0.16. Showing that
        as the evidence makes a wrong answer look inexplicable.
        """
        self.votes.append(m.employee_id)
        if m.employee_id is None:
            self.missed += 1
        else:
            self.identified += 1
            self.tally[m.employee_id] += 1
            cur = self.per_identity.get(m.employee_id)
            if cur is None or m.score > cur[0]:
                # The body crop is stored with the face from the SAME frame, so
                # the dashboard shows the person whose face decided the identity
                # at the moment it did - not a body from some other instant.
                self.per_identity[m.employee_id] = [
                    m.score, snapshot if snapshot is not None
                    else (cur[1] if cur is not None else None),
                    m.margin,
                    person if person is not None
                    else (cur[3] if cur is not None and len(cur) > 3 else None)]
        if snapshot is not None and quality > self.best_quality:
            self.best_quality = quality
            self.quality_snapshot = snapshot
        return self.provisional_id

    @property
    def provisional_id(self) -> int | None:
        """Current leader — for the LIVE overlay, never for attendance.

        Deliberately cheap to satisfy: the live view showing a name that later
        changes is a cosmetic wobble, whereas an attendance row naming the wrong
        person is a real error.  `finalize()` is the only thing attendance reads.
        """
        if not self.tally:
            return None
        emp, n = self.tally.most_common(1)[0]
        return emp if n >= self.required else None

    @property
    def agreement(self) -> float:
        """Leader's share of the identified frames. 0.0 when nothing matched."""
        if not self.tally or self.identified <= 0:
            return 0.0
        return self.tally.most_common(1)[0][1] / self.identified

    def finalize(self) -> int | None:
        """Decide the pass. Called once, when the tracker drops the person.

        Idempotent: the answer is cached, so a second call cannot change an
        identity that has already been written to attendance.
        """
        if self.finalized:
            return self.committed
        self.finalized = True
        self.committed = self._consensus_winner()
        return self.committed

    def _consensus_winner(self) -> int | None:
        if not self.tally or self.identified <= 0:
            return None
        emp, n = self.tally.most_common(1)[0]
        if n < self.min_recognitions:
            return None                      # too little evidence to judge
        if (n / self.identified) < self.consensus:
            return None                      # the pass disagreed with itself
        return emp

    def best_for(self, employee_id: int | None) -> tuple[float, np.ndarray | None]:
        """Best score and frame observed FOR ONE identity."""
        v = self.per_identity.get(employee_id)
        return (v[0], v[1]) if v is not None else (0.0, None)

    def _scope_id(self) -> int | None:
        """Whose evidence the `best_*` accessors describe.

        The final identity once decided; the current leader before that, so the
        live overlay never shows one person's name beside another's face.  A
        bare running maximum over all identities is exactly the bug fixed in
        99cf420 and must not come back.
        """
        return self.committed if self.committed is not None else self.provisional_id

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
        scope = self._scope_id()
        if scope is not None:
            return self.best_for(scope)[0]
        return self._global_best()[0]

    @property
    def best_snapshot(self) -> np.ndarray | None:
        """The highest-scoring frame FOR the committed identity.

        Before a commit there is no identity to scope to, so the running best
        is the only thing available; after one, the answer is definitionally
        the committed person's own best frame.
        """
        scope = self._scope_id()
        if scope is not None:
            snap = self.best_for(scope)[1]
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
        scope = self._scope_id()
        v = self.per_identity.get(scope) if scope is not None else None
        if v is None:
            g = self._global_best()
            return float(g[2]) if len(g) > 2 else 0.0
        return float(v[2])

    @property
    def best_person(self) -> np.ndarray | None:
        """Body crop from the frame that produced the committed identity's best
        face. This is what the dashboard shows: a person is recognisable to a
        human by build, clothing and gait, where a 112x112 aligned face is not.
        The face is kept alongside as the diagnostic image."""
        scope = self._scope_id()
        if scope is None:
            return None
        v = self.per_identity.get(scope)
        return v[3] if v is not None and len(v) > 3 else None

    @property
    def decided(self) -> bool:
        """True only after finalize() has named somebody.

        Nothing is decided mid-pass any more, so this is False for a track's
        whole life.  Recognition therefore continues for as long as the person
        is in view, which is the intent - see T1 in docs/RECOGNITION_PLAN.md.
        """
        return self.finalized and self.committed is not None
