# Direction of travel

## The problem

A camera's **role** says where it points, not which way a person was walking.
Both cameras here overlook the same lobby from opposite ends, so somebody
leaving the building walks through the entrance camera's field of view. Under
role-only logic that wrote a **check-in for a person who was going home**.

Camera identity cannot fix this. Direction has to come from the track.

## Two signals

**Tripwire crossing** — a line across the walkway with a declared "inside"
side. A track that starts on one side and ends on the other has crossed it,
and the side it ended on says which way. Primary signal, and what commercial
VMS platforms use.

**Depth trend** — these cameras look down a corridor, so a head walking toward
the lens grows and drifts down the frame; walking away it shrinks and rises.
Head-box area over the track is a real depth cue here, and it still works when
a track never reaches the tripwire.

Either alone can be fooled. Agreement is strong evidence; the line outranks
depth when they disagree, because a crossing is a geometric fact and depth is
an inference. A track with no usable signal returns UNKNOWN, and an UNKNOWN
track logs a sighting but changes no attendance state.

## The verdict is a property of the whole pass

Until 2026-09-06 the verdict was computed over a rolling window of the last
90 points (4.5 s at 20 fps) as "the last time the head changed sides", latched
for 5 s more, and then declared stale. The 2026-09-02..04 production export
showed two things wrong with that:

* **Pausing lost the verdict.** Someone who crossed the line and then stood
  still — at the door, at the desk, talking — had the crossing fall out of
  the window. 27 of the 41 recognised passes that produced no attendance in
  three days ended as `stale(6s, was ENTER)` and the like. The person had
  crossed; the system had forgotten.
* **A U-turn read as a crossing.** Somebody who walks to the door area and
  comes straight back ends where they began, but the last sign change in the
  window said ENTER, while the other camera — which saw only the outbound
  half — said EXIT. One walk, two confident opposite answers.

Now the track keeps every point from its first frame to its last (a numpy
array, decided in well under a millisecond however long the person stays), and
the question asked at the end is the only one attendance needs: *where did
this person end up, relative to where they started?*

| track | verdict | reason |
|---|---|---|
| started outside, ended inside | **ENTER** | `line+depth agree (ENTER)` / `line only` / `line=ENTER (depth said EXIT)` |
| started inside, ended outside | **EXIT** | likewise |
| both sides visited, ended where it began | **the side it ended on** | `crossed and returned (inside)` — the person went across and is demonstrably back |
| never crossed, moved far along the corridor, grew or shrank | depth verdict | `depth only (ENTER, travel 0.21)` |
| never crossed, moved sideways | UNKNOWN | `no crossing, sideways travel` — perspective alone changes a head's size |
| never crossed, barely moved | UNKNOWN | `no crossing, travel 0.07 < 0.15` |
| never got `min_travel` from its start | UNKNOWN | `stationary` |
| fewer than `min_points` | UNKNOWN | `only N points` |

There is no staleness any more: a crossing followed by ten minutes standing
still is still a crossing, because the person is still on the other side.

Details that matter:

* **Start and end sides** come from the first and last clean points — clean
  meaning further than `dead_band` (0.012 of the frame, ~26 px at 4K) from
  the line, so a head hovering on the line votes for neither side. The
  outermost point decides when three of the outermost five agree with it;
  otherwise the majority of ten does. A person who appears half a second
  before crossing keeps their starting side; one wild detection cannot move
  it.
* **Depth uses detected heads only**, and never a box clipped by the frame
  border: a synthesised head (from the person box, when the detector misses)
  is now sized from the last detected head, but the real ones are still the
  evidence, and a head passing under the camera loses half its box in the
  last frames for a reason that has nothing to do with distance.
* **A depth-only verdict must be borne out by motion along the corridor**:
  approaching means moving down the frame. A person walking sideways along the
  far end grew 3x through perspective and used to be called ENTER.

## How attendance uses it

`AttendanceService._effective_role()` maps direction onto meaning:

```
ENTER    -> counts as an IN  sighting, whichever camera saw it
EXIT     -> counts as an OUT sighting, whichever camera saw it
UNKNOWN  -> transition "NO_DIRECTION": event logged, state untouched
```

The camera's `role` is only a fallback, used when that camera has **no line
configured** (`require_direction=False`). Startup logs a warning for any camera
in that state.

## One walk, two cameras

Both cameras see every walk, and one of them often sees only half of it. The
arbiter (`app/services/arbiter.py`) holds a completed, identified pass for
`cross_camera_window_s` (15 s), and for as long as the same person is still
provisionally identified in a live track on either camera (up to four windows),
so the return leg of a U-turn joins the group instead of being decided
separately. Then two questions are answered on their own evidence:

* **Who** — the pass with the most agreeing frames, then the best score. Its
  snapshot, score and camera go on the event.
* **Which way** — a tripwire verdict (`line…` or `crossed and returned…`)
  outranks a depth-only one, which outranks none; among equally strong
  verdicts that disagree, the track that **ended last** wins, because the
  final movement is what decides where the person is now. The overruled view
  is named on the event: `line+depth agree (ENTER) [over Exit:EXIT, ended
  earlier]`.

Measured on the export by replaying its 579 passes (`bench/replay_events.py`):
the 14 contradicted groups all resolve to the later view, three inspected
U-turns that had checked people out while they were inside become
re-sightings, and the affected days recover 41–49 minutes of worked time each.

## Configuring a camera

```bash
python scripts/set_direction.py --show      # what is set
python scripts/set_direction.py --preview   # render each view with its line -> data/direction/

python scripts/set_direction.py --camera 1 --line 0.0,0.42,1.0,0.50 \
                                --inside below --depth grow
```

- `--line x1,y1,x2,y2` normalized 0..1, so it survives resolution changes
- `--inside above|below|left|right` — which side of the line is inside the building
- `--depth grow|shrink` — does a head grow or shrink as the person walks **inward**

The preview shades the inside half green. Check it before trusting the config.
Restart the server to apply.

## Current site configuration

| camera | role | line | inside | depth inward |
|---|---|---|---|---|
| 1 Entrance (192.168.1.2) | IN | 0.00,0.42 → 1.00,0.50 | below (foreground) | grow |
| 2 Exit (192.168.1.64) | OUT | 0.00,0.42 → 1.00,0.50 | above (far corridor) | shrink |

Both confirmed against the building layout: the corridor end of the lobby is
"inside", the door-and-reception end is "outside", and the two cameras face
each other — which is why they have opposite `inside` sides. Note what that
makes of the reception desk: it is on the outside half, so somebody standing
there has not entered yet, and somebody who walks from the corridor to the
desk has, for attendance, gone out.

## Tuning

| symptom | knob |
|---|---|
| too many NO_DIRECTION events | lower `min_travel` (default 0.06), or move the line to where people actually cross |
| directions occasionally inverted | check `--inside` and `--depth` in the preview; they must agree with the real layout |
| people counted while loitering | raise `min_travel` |
| short tracks never resolve | lower `min_points` (default 5), or raise `process_every_nth` frequency |
| a head hovering on the line flips sides | raise `dead_band` in `DirectionConfig` |

Every event stores `direction` and `direction_reason`, so a wrong call can be
audited rather than guessed at:

```sql
SELECT ts, direction, direction_reason, transition FROM recognition_event
ORDER BY ts DESC LIMIT 20;
```

## Measuring it

Three tools, cheapest first:

* `tests/test_direction_scenarios.py` — synthetic walks for every case in the
  table above, including the ones the export got wrong.
* `bench/dump_trajectories.py` replays recorded clips through the real
  pipeline and keeps every track's trajectory plus a frame strip for
  labelling; `bench/direction_eval.py` re-decides those trajectories with the
  current rule and the pre-2026-09-06 rule and scores both against
  `bench/direction_labels.json`. On 23 hand-labelled corridor tracks the old
  rule was right on 13, refused 9 and was wrong on 1; the current rule is
  right on all 23.
* `bench/replay_events.py` replays a production database's passes through
  the arbiter and the state machine and prints the old and new decisions side
  by side; `tests/test_attendance_replay.py` pins the inspected cases.
