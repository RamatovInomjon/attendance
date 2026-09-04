#!/usr/bin/env bash
# Restart EmAtSy safely: stop what is running, take the code that is on disk,
# migrate, and prove exactly one process came back on the right port.
#
#     scripts/restart.sh                # the normal case
#     scripts/restart.sh --stop         # stop and stay stopped
#     scripts/restart.sh --status       # what is running, change nothing
#     scripts/restart.sh --no-migrate   # skip the schema step
#     scripts/restart.sh --no-backup    # skip the database copy
#
# Everything here is a lesson from a restart that went wrong.
#
# IT FINDS THE PROCESS BY ITS WORKING DIRECTORY, never by a command pattern.
# gpu6 runs several unrelated projects, and `pkill -f "scripts/run.py"` has
# already killed an SSH session outright - the pattern matches the remote
# shell's own command string. /proc/<pid>/cwd cannot be confused that way, so
# this can only ever stop a process belonging to THIS checkout.
#
# IT PROVES THE STOP. SIGTERM is not always enough: a worker blocked on an RTSP
# read ignores it. Starting anyway gives you two processes on the same cameras
# and the same database - the second binds the port and looks healthy while the
# first keeps running the old code, and nothing in the UI reveals it.
#
# IT CLEARS __pycache__. app/core ships compiled .so modules; a cached .pyc from
# a previous version can shadow the module you just deployed, so the deploy
# appears to do nothing. This has cost hours twice.
#
# IT CHECKS BEFORE IT STARTS, not after. A CPU-only onnxruntime shadowing
# onnxruntime-gpu costs ~10x and starts perfectly happily, and a gallery built
# by a different recognizer now refuses to load at all. Both are far cheaper to
# see here than in a log after the cameras are down.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-}"
if [ -z "$PY" ]; then
    for c in "$ROOT/../venv/bin/python" "$HOME/faceid/venv/bin/python" \
             /home/inomjon/anaconda3/envs/yolo/bin/python3 "$(command -v python3)"; do
        [ -x "$c" ] && { PY="$c"; break; }
    done
fi
[ -x "$PY" ] || { echo "no interpreter found; set PY=/path/to/python" >&2; exit 1; }
PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"

LOG="${LOG:-$ROOT/run.log}"
PIDFILE="$ROOT/data/ematsy.pid"
DO_MIGRATE=1; DO_BACKUP=1; ACTION=restart
for a in "$@"; do case "$a" in
    --no-migrate) DO_MIGRATE=0 ;;
    --no-backup)  DO_BACKUP=0 ;;
    --stop)       ACTION=stop ;;
    --status)     ACTION=status ;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
esac; done

say() { printf '  %s\n' "$*"; }
die() { printf '\n  FAILED: %s\n' "$*" >&2; exit 1; }

PORT="$("$PY" -c 'from app.config import settings; print(settings.port)' 2>/dev/null || echo "")"
[ -n "$PORT" ] || die "cannot read app.config - wrong interpreter, or run from the wrong directory?"

# --- who is running -------------------------------------------------------
# A pid belongs to us when its working directory IS this checkout. Nothing
# about the command line is trusted, so no pattern can match a bystander.
ours() {
    local pid
    for pid in $(ps -eo pid= 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        [ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" = "$ROOT" ] || continue
        grep -qa 'scripts/run.py' "/proc/$pid/cmdline" 2>/dev/null && echo "$pid"
    done
}

listening() { ss -ltnH "sport = :$PORT" 2>/dev/null | grep -q . ; }

if [ "$ACTION" = status ]; then
    pids="$(ours)"
    if [ -n "$pids" ]; then
        say "running:"
        ps -o pid=,etime=,rss=,cmd= -p $(echo $pids | tr ' ' ',') | sed 's/^/    /'
    else
        say "not running"
    fi
    listening && say "port $PORT is bound" || say "port $PORT is free"
    exit 0
fi

# --- stop -----------------------------------------------------------------
echo
say "checkout    $ROOT"
say "interpreter $PY"
say "port        $PORT"
echo

pids="$(ours)"
if [ -z "$pids" ]; then
    say "nothing running from this directory"
else
    say "stopping: $(echo $pids | tr '\n' ' ')"
    kill $pids 2>/dev/null
    for _ in $(seq 1 30); do
        sleep 1
        [ -z "$(ours)" ] && break
    done
    left="$(ours)"
    if [ -n "$left" ]; then
        say "SIGTERM ignored (a worker blocked on an RTSP read does this) - forcing"
        kill -9 $left 2>/dev/null
        sleep 2
    fi
    [ -z "$(ours)" ] || die "still running: $(ours)"
    say "stopped"
fi
rm -f "$PIDFILE"

# The port outliving the process means something ELSE holds it - another
# checkout, or a stale container. Starting now would fail confusingly.
for _ in $(seq 1 10); do listening || break; sleep 1; done
listening && die "port $PORT still bound by something outside this checkout:
$(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"

[ "$ACTION" = stop ] && { say "stopped, staying stopped"; exit 0; }

# --- take the code that is on disk ----------------------------------------
find app scripts -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
say "cleared __pycache__ (a stale .pyc shadows the compiled .so)"

DB="$("$PY" -c '
from app.config import settings
u = settings.database_url_sync
print(u.split("///", 1)[1] if u.startswith("sqlite") and "///" in u else "")' 2>/dev/null || echo "")"
if [ "$DO_BACKUP" = 1 ] && [ -n "$DB" ] && [ -f "$DB" ]; then
    # .backup, never cp: the database is in WAL mode and a plain copy can tear.
    BK="${DB%.db}.pre-restart-$(date +%Y%m%d_%H%M).db"
    if command -v sqlite3 >/dev/null && sqlite3 "$DB" ".backup '$BK'" 2>/dev/null; then
        say "backed up  $(basename "$BK")"
    else
        "$PY" - "$DB" "$BK" <<'PYBK' && say "backed up  $(basename "$BK")"
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
with sqlite3.connect(src) as a, sqlite3.connect(dst) as b:
    a.backup(b)
PYBK
    fi
fi

if [ "$DO_MIGRATE" = 1 ]; then
    "$PY" scripts/migrate.py || die "migration failed - nothing was started"
fi

# --- pre-flight: the two things that are fatal at startup -----------------
"$PY" - <<'PYCHK' || die "pre-flight failed - nothing was started"
import sys
from app.config import settings
from app.core.onnx_env import available_providers

p = available_providers()
print(f"  providers   {p}")
if "CUDAExecutionProvider" not in p:
    print("  a CPU-only onnxruntime shadows onnxruntime-gpu and costs ~10x.\n"
          "  pip uninstall -y onnxruntime && pip install --force-reinstall "
          "--no-deps onnxruntime-gpu==1.23.2", file=sys.stderr)
    raise SystemExit(1)

# load_gallery() REFUSES a gallery built by a different recognizer, because
# comparing one model's vectors against another's succeeds silently and returns
# plausible nonsense. It is a hard failure at startup; find it here instead.
from app.services.enrollment import load_gallery
g = load_gallery()
print(f"  gallery     {len(g)} embeddings / {g.n_people} people")
floored = 0 if getattr(g, "_floor", None) is None else int((g._floor > 0).sum())
print(f"  floors      {floored} corridor crop(s) with their own threshold")
print(f"  threshold   {settings.threshold_for(settings.recognizer_model):.3f}")

# Not fatal - the live page falls back to MJPEG - but it must be VISIBLE.
# `pip install uvicorn` leaves this out; only `uvicorn[standard]` pulls it in,
# and without it every /ws/... handshake is served as ordinary HTTP and
# answered with a login redirect. The camera panels then sit on "Mavjud emas"
# with nothing in the log to say why, which is how it went unnoticed here.
from app.api.main import websocket_transport
ws = websocket_transport()
print(f"  websockets  {ws or 'MISSING - live view will use MJPEG'}")
if ws is None:
    print("  to restore the socket transport: pip install --no-deps websockets",
          file=sys.stderr)
PYCHK

# --- start ----------------------------------------------------------------
# setsid so it survives the SSH session that started it.
setsid nohup "$PY" scripts/run.py >>"$LOG" 2>&1 </dev/null &
sleep 1
say "starting, log -> $LOG"

for _ in $(seq 1 60); do
    listening && break
    sleep 1
done

pids="$(ours)"
n=$(echo "$pids" | grep -c . || true)
[ "$n" = 1 ] || die "expected exactly one process, found $n: $(echo $pids | tr '\n' ' ')
Two processes on the same cameras and database is the failure this guards."
echo "$pids" > "$PIDFILE"
listening || die "did not bind port $PORT within 60s. Last of $LOG:
$(tail -20 "$LOG")"

echo
say "running as pid $pids on port $PORT"
grep -aE "worker started|gallery:|recognizer |providers|reid |ERROR|Traceback" "$LOG" \
    | tail -12 | sed 's/^/    /'
echo
say "watch it:  tail -f $LOG"
say "stop it:   scripts/restart.sh --stop"
