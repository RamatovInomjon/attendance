"""One physical walk, one attendance decision - across both cameras.

Both cameras overlook the same corridor, so a single person walking past is
seen by both. Their direction geometry is mirrored (`inside_side` 1 vs -1,
`depth_grows_inward` 1 vs 0), so the SAME motion is labelled ENTER by one and
EXIT by the other, and `_effective_role` lets direction override the camera's
role. The debounce in `AttendanceService` is scoped to `camera_id`, so both
halves passed it and both were applied.

Measured on 2026-08-26: 7 contradictory pairs within 10 s, 25 within the 90 s
cooldown. Employee 9 checked in at 07:31:04 and out at 07:31:09 - a four-second
working day. Employee 6 got 72 seconds. Six rows ended with `check_out_time`
BEFORE `check_in_time`. Re-confirmed live on 2026-08-31: every one of eight
walk-throughs triggered both cameras, 2-4 s apart.

The fix is to decide the pass, not the sighting. A completed track is held for
`cross_camera_window_s`; anything else arriving for the same person in that
window belongs to the same walk. When the window closes the STRONGEST pass wins
and is the only one that moves attendance state - the others are recorded as
sightings so the evidence is not lost.

Strongest means most agreeing frames, then best score. Not "first", which is
what a plain person-scoped debounce would give: the first camera to finish its
track is an accident of when each person left each field of view, and the
weaker view is just as likely to win it. The frame count is what the identity
was actually decided on, so it is the honest measure of which view saw more.

Unknowns never come here. There is no identity to fuse on, and two cameras
seeing two strangers is two sightings, not one.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from app.config import settings


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


@dataclass
class PassGroup:
    winner: PendingPass
    others: list[PendingPass] = field(default_factory=list)


class PassArbiter:
    """Shared by every camera worker. Small, and guarded by one lock."""

    def __init__(self, window_s: float | None = None):
        self.window = (settings.cross_camera_window_s if window_s is None
                       else float(window_s))
        self._lock = threading.Lock()
        self._pending: dict[int, list[PendingPass]] = {}
        self.fused = 0          # passes suppressed as duplicates of another view

    def submit(self, p: PendingPass) -> None:
        with self._lock:
            self._pending.setdefault(p.employee_id, []).append(p)

    def due(self, now: float, *, force: bool = False) -> list[PassGroup]:
        """Groups whose window has closed. Each is popped exactly once, so two
        workers draining concurrently cannot both persist the same pass."""
        out: list[PassGroup] = []
        with self._lock:
            for emp in list(self._pending):
                group = self._pending[emp]
                oldest = min(x.monotonic for x in group)
                if not force and (now - oldest) < self.window:
                    continue
                del self._pending[emp]
                ranked = sorted(group, key=lambda x: x.strength, reverse=True)
                self.fused += len(ranked) - 1
                out.append(PassGroup(winner=ranked[0], others=ranked[1:]))
        return out

    def drain(self) -> list[PassGroup]:
        """Everything still held, regardless of window. For shutdown."""
        return self.due(0.0, force=True)

    def stats(self) -> dict:
        with self._lock:
            return {"pending": sum(len(v) for v in self._pending.values()),
                    "fused": self.fused, "window_s": self.window}
