# Direction of travel

## The problem

A camera's **role** says where it points, not which way a person was walking.
Both cameras here overlook the same open corridor, so somebody leaving the
building walks through the entrance camera's field of view. Under role-only
logic that wrote a **check-in for a person who was going home**.

Camera identity cannot fix this. Direction has to come from the track.

## Two signals

**Tripwire crossing** — a line across the walkway with a declared "inside"
side. A track that crosses it gives an unambiguous ENTER or EXIT. Primary
signal, and what commercial VMS platforms use.

**Depth trend** — these cameras look down a corridor, so a face walking toward
the lens grows and drifts down the frame; walking away it shrinks and rises.
Face-box area over the track is a real depth cue here, and it still works when a
track never reaches the tripwire.

Either alone can be fooled: someone loiters across a line, or turns round
mid-corridor. Agreement is strong evidence. **Disagreement returns UNKNOWN**,
and an UNKNOWN track logs a sighting but changes no attendance state.

Refusing to guess is deliberate. A wrong direction writes a wrong attendance
row; a missed one costs nothing, because the person is seen again on their next
pass.

## Decision table

| tripwire | depth | verdict |
|---|---|---|
| ENTER | ENTER | **ENTER** |
| EXIT | EXIT | **EXIT** |
| ENTER | EXIT | ENTER — the line is geometric fact, depth is inference |
| — | ENTER/EXIT, line configured | **UNKNOWN** — moved within one zone only |
| — | ENTER/EXIT, no line configured | depth verdict |
| travel < `min_travel` | any | **UNKNOWN** — standing still |
| fewer than `min_points` | any | **UNKNOWN** — track too short |

## How attendance uses it

`AttendanceService._effective_role()` maps direction onto meaning:

```
ENTER    -> counts as an IN  sighting, whichever camera saw it
EXIT     -> counts as an OUT sighting, whichever camera saw it
UNKNOWN  -> transition "NO_DIRECTION": event logged, state untouched
```

The camera's `role` is now only a fallback, used when that camera has **no line
configured** (`require_direction=False`). Startup logs a warning for any camera
in that state.

Verified in `bench/test_direction.py`:

```
 local cam role direction | transition    presence     in    out  worked
 09:00       IN     ENTER | CHECK_IN      INSIDE    09:00      -   0.00h
 12:00       IN      EXIT | CHECK_OUT     OUTSIDE   09:00  12:00   3.00h  <- was a false check-in
 13:00      OUT     ENTER | CHECK_IN      INSIDE    09:00  12:00   3.00h
 15:00       IN   UNKNOWN | NO_DIRECTION  INSIDE    09:00  12:00   3.00h  <- loitering, ignored
 18:00      OUT      EXIT | CHECK_OUT     OUTSIDE   09:00  18:00   8.00h
```

The 12:00 row is the reported bug: the entrance camera sees someone walking out,
and it is now a check-out.

## Configuring a camera

```bash
python scripts/set_direction.py --show      # what is set
python scripts/set_direction.py --preview   # render each view with its line -> data/direction/

python scripts/set_direction.py --camera 1 --line 0.0,0.42,1.0,0.50 \
                                --inside below --depth grow
```

- `--line x1,y1,x2,y2` normalized 0..1, so it survives resolution changes
- `--inside above|below|left|right` — which side of the line is inside the building
- `--depth grow|shrink` — does a face grow or shrink as the person walks **inward**

The preview shades the inside half green. Check it before trusting the config.
Restart the server to apply.

## Current site configuration

| camera | role | line | inside | depth inward |
|---|---|---|---|---|
| 1 Entrance (192.168.1.2) | IN | 0.00,0.42 → 1.00,0.50 | below (foreground) | grow |
| 2 Exit (192.168.1.64) | OUT | 0.00,0.42 → 1.00,0.50 | above (far corridor) | shrink |

Both confirmed against the building layout: arriving people walk *toward* the
entrance camera, and leaving people walk *toward* the exit camera — which is why
the two cameras have opposite `inside` sides.

## Tuning

| symptom | knob |
|---|---|
| too many NO_DIRECTION events | lower `min_travel` (default 0.06), or move the line to where people actually cross |
| directions occasionally inverted | check `--inside` and `--depth` in the preview; they must agree with the real layout |
| people counted while loitering | raise `min_travel` |
| short tracks never resolve | lower `min_points` (default 5), or raise `process_every_nth` frequency |

Every event stores `direction` and `direction_reason`, so a wrong call can be
audited rather than guessed at:

```sql
SELECT ts, direction, direction_reason, transition FROM recognition_event
ORDER BY ts DESC LIMIT 20;
```
