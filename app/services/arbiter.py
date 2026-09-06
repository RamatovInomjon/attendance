"""One physical walk, one attendance decision - across both cameras.

Both cameras overlook the same lobby from opposite ends, so a single person
walking past is seen by both. Their direction geometry is mirrored
(`inside_side` 1 vs -1, `depth_grows_inward` 1 vs 0), so the SAME motion is
labelled the same way by each - when each sees the whole of it.

Measured on 2026-08-26: 7 contradictory pairs within 10 s, 25 within the 90 s
cooldown. Employee 9 checked in at 07:31:04 and out at 07:31:09 - a four-second
working day. Employee 6 got 72 seconds. Six rows ended with `check_out_time`
BEFORE `check_in_time`. Re-confirmed live on 2026-08-31: every one of eight
walk-throughs triggered both cameras, 2-4 s apart.

The fix is to decide the pass, not the sighting. A completed track is held
until nothing more has arrived for that person for `cross_camera_window_s`;
anything arriving inside that window belongs to the same walk. Then the group
is decided ONCE, and only that decision moves attendance state - the other
views are recorded as sightings so the evidence is not lost.

Two separate questions are answered for a group, and they used to be one:

* **Who.** The pass with the most agreeing frames, then the best score, is
  the identity evidence: its snapshot, its score, its camera. Not "first",
  which is an accident of when each person left each field of view.

* **Which way.** Direction is NOT taken from that same pass. The 2026-09-02..04
  export held 14 groups whose two views were both "line+depth agree" and
  contradicted each other, and inspection showed why: the person walked to
  the door area and came straight back. The Exit camera saw the outbound
  half (EXIT) and lost them under the lens; the Entrance camera saw the
  return (ENTER). Letting the stronger FACE decide the DIRECTION was a coin
  toss, and when EXIT won a person who never left was checked out - one day's
  worked time was short by 49 minutes from exactly this.

  So direction is resolved on its own evidence: a tripwire-based verdict
  outranks a depth-only one, which outranks no verdict; and among equally
  strong verdicts that disagree, the track that ENDED LAST wins, because the
  final movement is what decides where the person is now. A track that saw
  the whole U-turn ("crossed and returned") is a tripwire verdict for the
  side the person came back to, and it beats a half-view of the same walk.

Unknowns never come here. There is no identity to fuse on, and two cameras
seeing two strangers is two sightings, not one.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

ENTER, EXIT, UNKNOWN = "ENTER", "EXIT", "UNKNOWN"


@dataclass
class PendingPass:
    """A completed, identified track waiting to be reconciled."""
    employee_id: int
    camera_id: int | None
    role: Any
    ts: Any                     # datetime, the moment the track completed
    monotonic: float            # for window arithmetic, immune to clock steps
    track: Any                  # CompletedTrack
    snapshot: str | None
    camera_name: str = ""

    @property
    def strength(self) -> tuple:
        """More agreeing frames first, then the better score."""
        return (int(self.track.embedded_frames), float(self.track.best_score))

    @property
    def verdict(self) -> str | None:
        """ENTER, EXIT, or None when the track had nothing to say."""
        d = str(getattr(self.track, "direction", UNKNOWN) or UNKNOWN)
        return d if d in (ENTER, EXIT) else None

    @property
    def confidence(self) -> int:
        """3 for a tripwire verdict (a crossing, or a crossing and return),
        2 for depth-only, 0 for none."""
        if self.verdict is None:
            return 0
        reason = str(getattr(self.track, "direction_reason", ""))
        if reason.startswith("line") or reason.startswith("crossed"):
            return 3
        return 2


@dataclass
class PassGroup:
    winner: PendingPass
    others: list[PendingPass] = field(default_factory=list)
    # The group's resolved direction. Usually the winner's own; different
    # when another view of the same walk had the better direction evidence.
    direction: str = UNKNOWN
    direction_reason: str = ""
    decided_by: PendingPass | None = None

    @property
    def contradicted(self) -> bool:
        return "[over " in self.direction_reason


def resolve_direction(passes: list[PendingPass]) -> tuple[str, str, PendingPass]:
    """Pick the group's direction from its best evidence.

    Returns (direction, reason, the pass that supplied it). The reason names
    an overruled contradiction so it can be audited from the event row.
    """
    best = max(p.confidence for p in passes)
    if best == 0:
        # Nothing had a verdict: keep whichever pass the caller ranks first.
        p = max(passes, key=lambda x: (x.strength, x.monotonic))
        return UNKNOWN, str(getattr(p.track, "direction_reason", "")), p
    top = sorted((p for p in passes if p.confidence == best), key=lambda p: p.monotonic)
    chosen = top[-1]                       # the view that ended last
    verdict = chosen.verdict
    reason = str(getattr(chosen.track, "direction_reason", ""))
    disagreeing = [p for p in top[:-1] if p.verdict != verdict]
    if disagreeing:
        other = disagreeing[-1]
        reason = (f"{reason} [over {other.camera_name or other.camera_id}:"
                  f"{other.verdict}, ended earlier]")
    return verdict, reason[:96], chosen


class PassArbiter:
    """Shared by every camera worker. Small, and guarded by one lock."""

    def __init__(self, window_s: float | None = None, max_hold_s: float | None = None):
        self.window = (settings.cross_camera_window_s if window_s is None
                       else float(window_s))
        # A group normally closes `window` after its oldest pass - the two
        # views of one straight walk complete 2-5 s apart. It stays open, up
        # to `max_hold`, while its person is still in view on either camera,
        # because that live track is the rest of the evidence.
        self.max_hold = (4.0 * self.window if max_hold_s is None else float(max_hold_s))
        self._lock = threading.Lock()
        self._pending: dict[int, list[PendingPass]] = {}
        # Who each camera currently has in view with a provisional identity,
        # and when it last said so. A group is not closed while its person is
        # still live somewhere: the other half of a walk - the return leg of a
        # U-turn, typically - is a track that is still running, and deciding
        # before it completes is deciding on half the evidence. Measured on
        # the export: the two halves completed 15-60 s apart in 55 cases,
        # which no fixed window short of a minute would have joined.
        self._live: dict[int | None, tuple[set, float]] = {}
        self.fused = 0          # passes suppressed as duplicates of another view
        self.contradictions = 0  # groups whose views disagreed on direction

    def submit(self, p: PendingPass) -> None:
        with self._lock:
            self._pending.setdefault(p.employee_id, []).append(p)

    def note_live(self, camera_id: int | None, employee_ids, now: float) -> None:
        """Called by each worker every processed frame with the people its
        live tracks provisionally name."""
        with self._lock:
            self._live[camera_id] = (set(employee_ids), now)

    def _live_now(self, now: float, stale_s: float = 3.0) -> set:
        out: set = set()
        for ids, at in self._live.values():
            if now - at <= stale_s:
                out |= ids
        return out

    def _decide(self, group: list[PendingPass]) -> PassGroup:
        ranked = sorted(group, key=lambda x: x.strength, reverse=True)
        direction, reason, by = resolve_direction(group)
        out = PassGroup(winner=ranked[0], others=ranked[1:],
                        direction=direction, direction_reason=reason, decided_by=by)
        self.fused += len(ranked) - 1
        if out.contradicted:
            self.contradictions += 1
        return out

    def due(self, now: float, *, force: bool = False) -> list[PassGroup]:
        """Groups whose window has closed. Each is popped exactly once, so two
        workers draining concurrently cannot both persist the same pass."""
        out: list[PassGroup] = []
        with self._lock:
            live = self._live_now(now) if not force else set()
            for emp in list(self._pending):
                group = self._pending[emp]
                oldest = min(x.monotonic for x in group)
                if not force and (now - oldest) < self.max_hold:
                    if (now - oldest) < self.window:
                        continue
                    if emp in live:
                        continue          # still in view somewhere: wait for that track
                del self._pending[emp]
                out.append(self._decide(group))
        return out

    def drain(self) -> list[PassGroup]:
        """Everything still held, regardless of window. For shutdown."""
        return self.due(0.0, force=True)

    def stats(self) -> dict:
        with self._lock:
            return {"pending": sum(len(v) for v in self._pending.values()),
                    "fused": self.fused, "contradictions": self.contradictions,
                    "window_s": self.window}
